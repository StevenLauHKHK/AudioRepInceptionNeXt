import os
import pandas as pd
import pickle
import torch
import torch.utils.data

import slowfast.utils.logging as logging

from .build import DATASET_REGISTRY

from .spec_augment import combined_transforms
from . import utils as utils
from .audio_loader_nsynth import pack_audio

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Nsynth(torch.utils.data.Dataset):

    def __init__(self, cfg, mode):

        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Nsynth".format(mode)
        self.cfg = cfg
        self.mode = mode
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS

        logger.info("Constructing Nsynth {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the audio loader.
        """
        if self.mode == "train":
            path_annotations_pickle = os.path.join(self.cfg.NSYNTH.ANNOTATIONS_DIR, self.cfg.NSYNTH.TRAIN_LIST)
            self.data_dir = self.cfg.NSYNTH.TRAIN_AUDIO_DATA_DIR
        elif self.mode == "val":
            path_annotations_pickle = os.path.join(self.cfg.NSYNTH.ANNOTATIONS_DIR, self.cfg.NSYNTH.VAL_LIST)
            self.data_dir = self.cfg.NSYNTH.VALID_AUDIO_DATA_DIR
        else:
            path_annotations_pickle = os.path.join(self.cfg.NSYNTH.ANNOTATIONS_DIR, self.cfg.NSYNTH.TEST_LIST)
            self.data_dir = self.cfg.NSYNTH.TEST_AUDIO_DATA_DIR

        assert os.path.exists(path_annotations_pickle), "{} dir not found".format(
            path_annotations_pickle
        )

        self._audio_records = []
        self._temporal_idx = []
        for tup in pd.read_pickle(path_annotations_pickle).iterrows():
            for idx in range(self._num_clips):
                self._audio_records.append(tup[1])
                self._temporal_idx.append(idx)
        assert (
                len(self._audio_records) > 0
        ), "Failed to load Nsynth split {} from {}".format(
            self.mode, path_annotations_pickle
        )
        logger.info(
            "Constructing nsynth dataloader (size: {}) from {}".format(
                len(self._audio_records), path_annotations_pickle
            )
        )

    def __getitem__(self, index):
        """
        Given the audio index, return the spectrogram, label, and audio
        index.
        Args:
            index (int): the audio index provided by the pytorch sampler.
        Returns:
            spectrogram (tensor): the spectrogram sampled from the audio. The dimension
                is `channel` x `num frames` x `num frequencies`.
            label (int): the label of the current audio.
            index (int): Return the index of the audio.
        """

        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
        elif self.mode in ["test"]:
            temporal_sample_index = self._temporal_idx[index]
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        spectrogram = pack_audio(self.cfg, self._audio_records[index], temporal_sample_index, self.data_dir)
        # Normalization.
        spectrogram = spectrogram.float()
        if self.mode in ["train"]:
            # Data augmentation.
            # C T F -> C F T
            spectrogram = spectrogram.permute(0, 2, 1)
            # SpecAugment
            spectrogram = combined_transforms(spectrogram)
            # C F T -> C T F
            spectrogram = spectrogram.permute(0, 2, 1)
        label = self._audio_records[index]['class_id']
        spectrogram = utils.pack_pathway_output(self.cfg, spectrogram)

        return spectrogram, label, index, self._audio_records[index]['video']

    def __len__(self):
        return len(self._audio_records)