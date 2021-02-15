# The MIT License (MIT)
#
# Copyright (c) 2017-2021 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
from itertools import combinations
from typing import List, Optional, Tuple, Union

import numpy as np
import scipy.special
import torch
import torch.nn.functional as F
from scipy.cluster.hierarchy import fcluster

from pyannote.audio import Inference
from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.pipeline import Pipeline
from pyannote.audio.pipelines.utils import PipelineModel, get_devices, get_model
from pyannote.audio.utils.permutation import permutate
from pyannote.audio.utils.signal import Binarize
from pyannote.core import Annotation, SlidingWindow, SlidingWindowFeature
from pyannote.core.utils.hierarchy import pool
from pyannote.database import get_annotated
from pyannote.metrics.diarization import (
    DiarizationPurityCoverageFMeasure,
    GreedyDiarizationErrorRate,
)
from pyannote.pipeline.parameter import LogUniform, Uniform


class SpeakerDiarization(Pipeline):
    """Speaker diarization pipeline

    Parameters
    ----------
    segmentation : Model or str, optional
        `Inference` instance used to extract raw segmentation scores.
        When `str`, assumes that file already contains a corresponding key with
        precomputed scores. Defaults to "seg".
    embeddings : Inference or str, optional
        `Inference` instance used to extract speaker embeddings. When `str`,
        assumes that file already contains a corresponding key with precomputed
        embeddings. Defaults to "emb".
    metric : {'euclidean', 'cosine', 'angular'}, optional
        Metric used for comparing embeddings. Defaults to 'cosine'.
    purity : float, optional
        Optimize coverage for target purity.
        Defaults to optimizing diarization error rate.
    coverage : float, optional
        Optimize purity for target coverage.
        Defaults to optimizing diarization error rate.
    fscore : bool, optional
        Optimize for purity/coverage fscore.
        Defaults to optimizing for diarization error rate.

    # TODO: investigate the use of non-weighted fscore/purity/coverage

    """

    def __init__(
        self,
        segmentation: PipelineModel = "pyannote/Segmentation-PyanNet-DIHARD",
        embedding: PipelineModel = "hbredin/SpeakerEmbedding-XVectorMFCC-VoxCeleb",
        metric: Optional[str] = "cosine",
        purity: Optional[float] = None,
        coverage: Optional[float] = None,
        fscore: bool = False,
    ):

        super().__init__()

        segmentation = get_model(segmentation)
        self.embedding = get_model(embedding)

        chunk_duration = segmentation.specifications.duration
        segmentation_device, embedding_device = get_devices(needs=2)

        self.step = 0.25
        self.segmentation = Inference(
            segmentation,
            window="sliding",
            skip_aggregation=True,
            duration=chunk_duration,
            step=self.step * chunk_duration,
            batch_size=32,
            device=segmentation_device,
            progress_hook="Segmentation...",
        )

        self.embedding.to(embedding_device)
        self.metric = metric

        if sum((purity is not None, coverage is not None, fscore)):
            raise ValueError(
                "One must choose between optimizing for f-score, target purity, "
                "or target coverage."
            )

        self.purity = purity
        self.coverage = coverage
        self.fscore = fscore

        self.activity_threshold = Uniform(0.0, 1.0)

        self.activity_bias = Uniform(-1.0, 1.0)
        self.activity_slope = LogUniform(1e-2, 1e2)

        self.confidence_bias = Uniform(-1.0, 1.0)
        self.confidence_slope = LogUniform(1e-2, 1e2)

        self.consistency_threshold = Uniform(0.0, 1.0)
        self.clustering_threshold = Uniform(0.0, 2.0)

        self._binarize = Binarize(
            onset=0.5, offset=0.5, min_duration_on=0.0, min_duration_off=0.0
        )

    def apply(self, file: AudioFile) -> Annotation:
        """Apply speaker diarization

        Parameters
        ----------
        file : AudioFile
            Processed file.

        Returns
        -------
        diarization : Annotation
            Speaker diarization
        """

        num_speakers = len(self.segmentation.model.specifications.classes)

        # output of segmentation model on each chunk
        segmentations: SlidingWindowFeature = self.segmentation(file)
        num_frames_in_chunk = segmentations.data.shape[1]
        num_frames_in_step = round(self.step * num_frames_in_chunk)

        # confidence of segmentation model on each chunk
        # shape = (num_chunks, )
        confidences = np.mean(
            np.min((np.abs(segmentations.data - 0.5) / 0.5), axis=2), axis=1
        )

        # ratio of chunk during which each speaker is active
        # shape = (num_chunks, num_speakers)
        activities = np.mean(segmentations.data, axis=1)

        embeddings = []
        chunk_indices = []
        active_speaker_indices = []

        cannot_link: List[Tuple[int, int]] = []
        must_link = []
        k = 0

        previous_c = -np.inf
        previous_a = None

        audio = self.segmentation.model.audio

        for c, ((chunk, segmentation), activity, confidence) in enumerate(
            zip(segmentations, activities, confidences)
        ):

            # determine active speakers
            is_active = activity > self.activity_threshold
            num_active = np.sum(is_active)

            # skip chunk when there is no active speaker
            if num_active < 1:
                continue

            # load audio chunk
            waveform = audio.crop(file, chunk)[0].unsqueeze(dim=0)
            # shape (1, num_channels == 1, num_samples == chunk_duration x sample_rate)

            # extract embeddings of active speakers
            with torch.no_grad():

                # TODO: give more weights to non-overlapping regions
                # TODO: give way more weights to high-confidence regions
                # TODO: give more weights in the middle of chunks
                weights = torch.tensor(segmentation).T
                # shape (num_speakers, num_frames)

                embedding = self.embedding(
                    waveform.repeat(num_speakers, 1, 1).to(self.embedding.device),
                    weights=weights.to(self.embedding.device),
                )[is_active]
                # shape (num_active_speakers, emb_dimension)

            # scale embeddings so that high activity speakers & high confidence chunks
            # have longer norms. hence they will weigh more in the subsequent agglomerative
            # clustering with "pool" linkage.
            activity_scale = scipy.special.expit(
                (activity[is_active] - self.activity_bias) * self.activity_slope
            )
            # shape (num_active_speakers, )

            confidence_scale = scipy.special.expit(
                (confidence - self.confidence_bias) * self.confidence_slope
            )
            # shape (, )

            scale = np.sqrt(activity_scale * confidence_scale)
            # (num_active_speakers, )

            scaled_embedding = (
                scale[:, np.newaxis] * F.normalize(embedding, p=2, dim=1).cpu().numpy()
            )
            # shape (num_active_speakers, emb_dimension)

            embeddings.append(scaled_embedding)

            # keep track of chunk and active speaker indices
            # for post-clustering reconstruction
            a = np.where(is_active)[0]
            active_speaker_indices.append(a)
            chunk_indices.append(num_active * [c])

            # cannot-link constraints prevent merging two speakers
            # from the same chunk
            for (i, j) in combinations(range(num_active), 2):
                cannot_link.append((k + i, k + j))

            # must-link constraints force merging speakers who
            # speaks over two consecutive chunks
            if c == previous_c + 1:

                # find optimal mapping between active speakers of previous and current chunks
                _, (permutation,), (cost,) = permutate(
                    segmentations[c - 1][:-num_frames_in_step][np.newaxis, :, :],
                    segmentation[num_frames_in_step:],
                    returns_cost=True,
                )

                # permutation[i] == j indicates that jth speaker of current chunk
                # is mapped to ith speaker of previous chunk.

                for i, j in enumerate(permutation):
                    try:
                        i_ = list(previous_a).index(i)
                        j_ = list(a).index(j)
                        if cost[i, j] < self.consistency_threshold:
                            must_link.append((k + j_, k - len(previous_a) + i_))
                    except ValueError:
                        pass

            previous_a = a
            previous_c = c
            k += num_active

        if not embeddings:
            return Annotation(uri=file["uri"])

        embeddings = np.vstack(embeddings)
        chunk_indices = np.hstack(chunk_indices)
        active_speaker_indices = np.hstack(active_speaker_indices)

        # hierarchical agglomerative clustering with "pool" linkage
        Z = pool(
            embeddings,
            metric=self.metric,
            cannot_link=cannot_link,
            must_link=must_link,
            must_link_method="merge",
        )
        clusters = fcluster(Z, self.clustering_threshold, criterion="distance")

        num_clusters = len(np.unique(clusters))
        frame_duration = (
            self.segmentation.model.introspection.inc_num_samples / audio.sample_rate
        )
        num_frames_in_file = math.ceil(audio.get_duration(file) / frame_duration)
        aggregated = np.zeros((num_frames_in_file, num_clusters))
        overlapped = np.zeros((num_frames_in_file, num_clusters))

        chunks = segmentations.sliding_window
        num_frames_in_chunk = segmentations.data.shape[1]

        for cluster, c, a in zip(clusters, chunk_indices, active_speaker_indices):
            start_frame = round(chunks[c].start / frame_duration)
            # TODO: give more weights to the middle of chunks
            aggregated[
                start_frame : start_frame + num_frames_in_chunk, cluster - 1
            ] += (segmentations.data[c, :, a] * confidences[c])
            overlapped[
                start_frame : start_frame + num_frames_in_chunk, cluster - 1
            ] += confidences[c]

        frames = SlidingWindow(start=0.0, step=frame_duration, duration=frame_duration)
        speaker_probabilities = SlidingWindowFeature(
            aggregated / (overlapped + 1e-12), frames
        )

        diarization = Annotation(uri=file["uri"])
        for i, data in enumerate(speaker_probabilities.data.T):
            speaker_probability = SlidingWindowFeature(data.reshape(-1, 1), frames)
            for speaker_turn in self._binarize(speaker_probability):
                diarization[speaker_turn, i] = i

        diarization.speaker_probabilities = speaker_probabilities

        return diarization

    def loss(self, file: AudioFile, hypothesis: Annotation) -> float:
        """Compute coverage at target purity (or vice versa)

        Parameters
        ----------
        file : `dict`
            File as provided by a pyannote.database protocol.
        hypothesis : `pyannote.core.Annotation`
            Speech turns.

        Returns
        -------
        coverage (or purity) : float
            When optimizing for target purity:
                If purity < target_purity, returns (purity - target_purity).
                If purity > target_purity, returns coverage.
            When optimizing for target coverage:
                If coverage < target_coverage, returns (coverage - target_coverage).
                If coverage > target_coverage, returns purity.
        """

        fmeasure = DiarizationPurityCoverageFMeasure()

        reference: Annotation = file["annotation"]
        _ = fmeasure(reference, hypothesis, uem=get_annotated(file))
        purity, coverage, _ = fmeasure.compute_metrics()

        if self.purity is not None:
            if purity > self.purity:
                return purity - self.purity
            else:
                return coverage

        elif self.coverage is not None:
            if coverage > self.coverage:
                return coverage - self.coverage
            else:
                return purity

    def get_metric(
        self,
    ) -> Union[GreedyDiarizationErrorRate, DiarizationPurityCoverageFMeasure]:
        """Return new instance of diarization metric"""

        if (self.purity is not None) or (self.coverage is not None):
            raise NotImplementedError(
                "pyannote.pipeline will use `loss` method fallback."
            )

        if self.fscore:
            return DiarizationPurityCoverageFMeasure(collar=0.0, skip_overlap=False)

        # defaults to optimizing diarization error rate
        return GreedyDiarizationErrorRate(collar=0.0, skip_overlap=False)

    def get_direction(self):
        """Optimization direction"""

        if self.purity is not None:
            # we maximize coverage at target purity
            return "maximize"
        elif self.coverage is not None:
            # we maximize purity at target coverage
            return "maximize"
        elif self.fscore:
            # we maximize purity/coverage f-score
            return "maximize"
        else:
            # we minimize diarization error rate
            return "minimize"
