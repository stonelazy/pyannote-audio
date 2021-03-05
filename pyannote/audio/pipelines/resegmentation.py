# MIT License
#
# Copyright (c) 2018-2021 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Resegmentation pipeline"""

import math
import tempfile
from copy import deepcopy
from types import MethodType
from typing import List, Text

import numpy as np
import scipy.optimize
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ProgressBar
from torch.optim import SGD
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform

from pyannote.audio import Inference
from pyannote.audio.core.callback import GraduallyUnfreeze
from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.pipeline import Pipeline
from pyannote.audio.pipelines.utils import (
    PipelineAugmentation,
    PipelineModel,
    get_augmentation,
    get_devices,
    get_model,
)
from pyannote.audio.tasks import SpeakerTracking
from pyannote.audio.utils.signal import Binarize
from pyannote.core import Annotation, SlidingWindow, SlidingWindowFeature
from pyannote.database.protocol import SpeakerDiarizationProtocol
from pyannote.metrics.diarization import GreedyDiarizationErrorRate
from pyannote.pipeline.parameter import Categorical, Integer, LogUniform


class Resegmentation(Pipeline):
    """Self-supervised resegmentation (aka AdaptiveSpeakerTracking)

    Let M be a pretrained segmentation model.

    For each file f, this pipeline uses an initial speaker diarization result as a first
    step of speaker tracking labels.

    Those (automatic and therefore imprecise) labels are then used to do transfer learning
    from a pretrained segmentation model M into a speaker tracking model M_f, in a
    self-supervised manner.

    Finally, the fine-tuned model M_f is applied to file f to obtain the final (and
    hopefully better) speaker tracking labels.

    During transfer-learning, it is possible to weight some frames more than others: the
    intuition is that the model will use high confidence regions to learn speaker models
    and hence will eventually be able to correctly assign parts where the confidence
    was initially low.

    Conversely, to avoid overfitting too much to those high confidence regions, we use
    data augmentation and gradually unfreeze layers of the pretrained model M.

    Parameters
    ----------
    segmentation : Model, str, or dict, optional
        Pretrained segmentation model.
        Defaults to "pyannote/Segmentation-PyanNet-DIHARD".
    layers : list, optional
        Only fine-tune those layers, unfreezing them in that order.
        Defaults to fine-tuning all layers from output layer to input layer.
    augmentation : BaseWaveformTransform, or dict, optional
        torch_audiomentations waveform transform, used during fine-tuning.
        Defaults to no augmentation.
    diarization : str, optional
        File key to use as input diarization. Defaults to "diarization".
    confidence : str, optional
        File key to use as confidence. Defaults to not use any confidence estimation.
    verbose : bool, optional


    Hyper-parameters
    ----------------
    num_epochs : int
        Number of epochs (where one epoch = going through the file once) between
        each gradual unfreezing step.
    batch_size : int
        Batch size.
    learning_rate : float
        Learning rate.

    See also
    --------
    pyannote.audio.pipelines.utils.get_inference
    """

    def __init__(
        self,
        segmentation: PipelineModel = "pyannote/Segmentation-PyanNet-DIHARD",
        layers: List[Text] = None,
        augmentation: PipelineAugmentation = None,
        diarization: Text = "diarization",
        confidence: Text = None,
        verbose: bool = False,
    ):
        super().__init__()

        self.segmentation = segmentation
        self.layers = layers
        self.augmentation: BaseWaveformTransform = get_augmentation(augmentation)
        self.diarization = diarization
        self.confidence = confidence
        self.verbose = verbose

        # base pretrained segmentation model
        self.seg_model_ = get_model(segmentation)
        self.chunk_duration_ = self.seg_model_.specifications.duration
        self.audio_ = self.seg_model_.audio

        # if model is on CPU and GPU is available, move to GPU
        # if model is already on GPU, leave it there
        if self.seg_model_.device.type == "cpu" and torch.cuda.is_available():
            (device,) = get_devices(needs=1)
            self.seg_model_.to(device)

        # will be used to go from speaker activations SlidingWindowFeature instance
        # to an actual diarization (as Annotation instance).
        self.binarize_ = Binarize(
            onset=0.5,
            offset=0.5,
            min_duration_on=0.0,
            min_duration_off=0.0,
            pad_onset=0.0,
            pad_offset=0.0,
        )

        # hyper-parameters
        self.batch_size = Categorical([1, 2, 4, 8, 16, 32])
        self.epochs_per_layer = Integer(1, 20)
        self.learning_rate = LogUniform(1e-4, 1)

    def apply(self, file: AudioFile) -> Annotation:

        # do not fine tune the model if num_epochs is zero
        if self.epochs_per_layer == 0:
            return file[self.diarization]

        # create a copy of file
        file_copy = dict(file)

        # create a dummy train-only protocol where `file` is the only training file
        file_copy["annotation"] = file_copy[self.diarization]

        class DummyProtocol(SpeakerDiarizationProtocol):
            name = "DummyProtocol"

            # TODO: support multiple version of the same file? (e.g. )
            # TODO: support multi-file segmentation (e.g. for cross-show diarization)
            def train_iter(self):
                yield file_copy

            # TODO: support validation?

        spk = SpeakerTracking(
            DummyProtocol(),
            duration=self.chunk_duration_,
            balance=None,
            weight=self.confidence,
            batch_size=self.batch_size,
            num_workers=None,
            pin_memory=False,
            augmentation=self.augmentation,
        )

        fine_tuning_callback = GraduallyUnfreeze(
            schedule=self.layers, epochs_per_stage=self.epochs_per_layer
        )

        # duplicate the segmentation model as we will use it later
        speaker_tracking_model = deepcopy(self.seg_model_)
        speaker_tracking_model.task = spk

        def configure_optimizers(model):
            return SGD(model.parameters(), lr=self.learning_rate)

        speaker_tracking_model.configure_optimizers = MethodType(
            configure_optimizers, speaker_tracking_model
        )

        # TODO: add option to pass a directory to `apply` and have the trainer
        # use this directory and save logs in there. that would be useful for
        # debugging purposes. for now, a new temporary directory is created
        # on-the-fly and automatically destroyed after training.
        # TODO: this option might also activate progress bar and weights summary
        with tempfile.TemporaryDirectory() as default_root_dir:
            trainer = Trainer(
                gpus=1 if self.seg_model_.device.type == "cuda" else 0,
                callbacks=[
                    fine_tuning_callback,
                    ProgressBar(refresh_rate=1 if self.verbose else 0),
                ],
                checkpoint_callback=False,
                weights_summary="top" if self.verbose else None,
                default_root_dir=default_root_dir,
            )
            trainer.fit(speaker_tracking_model)

        segmentation_inference = Inference(
            self.seg_model_,
            window="sliding",
            skip_aggregation=True,
            duration=self.chunk_duration_,
            step=0.1 * self.chunk_duration_,
        )

        speaker_tracking_inference = Inference(
            speaker_tracking_model,
            window="sliding",
            skip_aggregation=True,
            duration=self.chunk_duration_,
            step=0.1 * self.chunk_duration_,
        )

        speakers = speaker_tracking_inference(file_copy)
        segmentations = segmentation_inference(file_copy)

        _, num_frames_in_chunk, num_speakers = speakers.data.shape
        frame_duration = (
            self.seg_model_.introspection.inc_num_samples / self.audio_.sample_rate
        )
        num_frames_in_file = math.ceil(
            self.audio_.get_duration(file_copy) / frame_duration
        )
        aggregated = np.zeros((num_frames_in_file, num_speakers))
        overlapped = np.zeros((num_frames_in_file, num_speakers))

        for (chunk, segmentation), (same_chunk, speaker) in zip(
            segmentations, speakers
        ):
            assert chunk == same_chunk
            start_frame = round(chunk.start / frame_duration)

            cost = 1.0 - np.einsum("nS,ns->Ss", speaker, segmentation) / np.einsum(
                "ns->s", segmentation
            )
            S, s = scipy.optimize.linear_sum_assignment(cost)
            for Si, si in zip(S, s):
                aggregated[
                    start_frame : start_frame + num_frames_in_chunk, Si
                ] += segmentation[:, si]
                overlapped[start_frame : start_frame + num_frames_in_chunk, Si] += 1

        frames = SlidingWindow(start=0.0, step=frame_duration, duration=frame_duration)

        speaker_probabilities = SlidingWindowFeature(
            aggregated / (overlapped + 1e-12), frames
        )

        file["debug/resegmentation/speaker_probabilities"] = speaker_probabilities

        return self.binarize_(speaker_probabilities)

    def get_metric(self) -> GreedyDiarizationErrorRate:
        return GreedyDiarizationErrorRate(collar=0.0, skip_overlap=False)
