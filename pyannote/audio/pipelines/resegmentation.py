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
from typing import Text

import numpy as np
import scipy.optimize
from pytorch_lightning import Trainer
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.callbacks import ProgressBar

from torch.optim import SGD
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform

from pyannote.audio import Inference
from pyannote.audio.core.callback import GraduallyUnfreeze
from pyannote.audio.core.io import AudioFile
from pyannote.audio.core.pipeline import Pipeline
from pyannote.audio.pipelines.utils import (
    PipelineAugmentation,
    PipelineInference,
    get_augmentation,
    get_inference,
)
from pyannote.audio.tasks import SpeakerTracking
from pyannote.core import Annotation, SlidingWindow, SlidingWindowFeature
from pyannote.database.protocol import SpeakerDiarizationProtocol
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
    segmentation : Inference, Model, str, or dict, optional
        Pretrained segmentation inference.
        Defaults to "pyannote/Segmentation-PyanNet-DIHARD".
    augmentation : BaseWaveformTransform, or dict, optional
        torch_audiomentations waveform transform, used during fine-tuning.
        Defaults to no augmentation.
    diarization : str, optional
        File key to use as input diarization. Defaults to "diarization".
    confidence : str, optional
        File key to use as confidence. Defaults to not use any confidence estimation.

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
        segmentation: PipelineInference = "pyannote/Segmentation-PyanNet-DIHARD",
        augmentation: PipelineAugmentation = None,
        diarization: Text = "diarization",
        confidence: Text = None,
    ):
        super().__init__()

        # base pretrained segmentation model
        self.segmentation: Inference = get_inference(segmentation)
        self.augmentation: BaseWaveformTransform = get_augmentation(augmentation)

        self.diarization = diarization
        self.confidence = confidence

        self.batch_size = Categorical([4, 8, 16, 32])
        self.num_epochs_per_layer = Integer(1, 20)
        self.learning_rate = LogUniform(1e-4, 1)

    def apply(self, file: AudioFile) -> Annotation:

        # create a copy of file
        file = dict(file)

        # do not fine tune the model if num_epochs is zero
        if self.num_epochs_per_layer == 0:
            return file[self.diarization]

        # create a dummy train-only protocol where `file` is the only training file
        file["annotation"] = file[self.diarization]

        class DummyProtocol(SpeakerDiarizationProtocol):
            name = "DummyProtocol"

            def train_iter(self):
                yield file

        spk = SpeakerTracking(
            DummyProtocol(),
            duration=self.segmentation.duration,
            balance=None,
            weight=self.confidence,
            batch_size=self.batch_size,
            num_workers=None,
            pin_memory=False,
            augmentation=self.augmentation,
        )

        fine_tuning_callback = GraduallyUnfreeze(patience=self.num_epochs_per_layer)
        max_epochs = (
            len(ModelSummary(self.segmentation.model, mode="top").named_modules)
            * self.num_epochs_per_layer
        )

        # duplicate the segmentation model as we will use it later
        speaker_tracking_model = deepcopy(self.segmentation.model)
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
                max_epochs=max_epochs,
                gpus=1 if self.segmentation.device.type == "cuda" else 0,
                callbacks=[fine_tuning_callback, ProgressBar(refresh_rate=0)],
                checkpoint_callback=False,
                weights_summary=None,
                default_root_dir=default_root_dir,
            )
            trainer.fit(speaker_tracking_model)

        segmentation = Inference(
            self.segmentation.model,
            window="sliding",
            skip_aggregation=True,
            duration=self.segmentation.duration,
            step=0.1 * self.segmentation.duration,
            batch_size=self.segmentation.batch_size,
            device=self.segmentation.device,
        )

        speaker_tracking = Inference(
            speaker_tracking_model,
            window="sliding",
            skip_aggregation=True,
            duration=self.segmentation.duration,
            step=0.1 * self.segmentation.duration,
            batch_size=self.segmentation.batch_size,
            device=self.segmentation.device,
        )

        speakers = speaker_tracking(file)
        segmentations = segmentation(file)
        audio = self.segmentation.model.audio

        _, num_frames_in_chunk, num_speakers = speakers.data.shape
        frame_duration = (
            segmentation.model.introspection.inc_num_samples / audio.sample_rate
        )
        num_frames_in_file = math.ceil(audio.get_duration(file) / frame_duration)
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
        return speaker_probabilities
