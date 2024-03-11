import os
import pandas as pd
import pickle
import torch
import h5py
import torch.utils.data
from fvcore.common.file_io import PathManager

import slowfast.utils.logging as logging

from .build import DATASET_REGISTRY

from .spec_augment import combined_transforms
from . import utils as utils
from .audio_loader_urbansound8k import pack_audio

logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Urbansound8k(torch.utils.data.Dataset):

    def __init__(self, cfg, mode):

        assert mode in [
            "train",
            "val",
            "test",
            "train+val"
        ], "Split '{}' not supported for UrbanSound8K".format(mode)
        self.cfg = cfg
        self.mode = mode
        if self.mode in ["train", "val", "train+val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = cfg.TEST.NUM_ENSEMBLE_VIEWS

        self.audio_dataset = None
        logger.info("Constructing UrbanSound8K Audio {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the audio loader.
        """
        if self.mode == "train":
            path_annotations_pickle = [os.path.join(self.cfg.URBANSOUND8K.ANNOTATIONS_DIR, self.cfg.URBANSOUND8K.TRAIN_LIST)]
        elif self.mode == "val":
            path_annotations_pickle = [os.path.join(self.cfg.URBANSOUND8K.ANNOTATIONS_DIR, self.cfg.URBANSOUND8K.VAL_LIST)]
        elif self.mode == "test":
            path_annotations_pickle = [os.path.join(self.cfg.URBANSOUND8K.ANNOTATIONS_DIR, self.cfg.URBANSOUND8K.TEST_LIST)]
        else:
            path_annotations_pickle = [os.path.join(self.cfg.URBANSOUND8K.ANNOTATIONS_DIR, file)
                                       for file in [self.cfg.URBANSOUND8K.TRAIN_LIST, self.cfg.URBANSOUND8K.VAL_LIST]]

        for file in path_annotations_pickle:
            assert PathManager.exists(file), "{} dir not found".format(
                file
            )

        self._audio_records = []
        self._temporal_idx = []
        for file in path_annotations_pickle:
            for tup in pd.read_pickle(file).iterrows():
                for idx in range(self._num_clips):
                    self._audio_records.append(tup[1])
                    self._temporal_idx.append(idx)
        assert (
                len(self._audio_records) > 0
        ), "Failed to load Speech Commnd v2 split {} from {}".format(
            self.mode, path_annotations_pickle
        )
        logger.info(
            "Constructing Speech Commnd v2 dataloader (size: {}) from {}".format(
                len(self._audio_records), path_annotations_pickle
            )
        )

    def __getitem__(self, index):
        """
        Given the audio index, return the spectrogram, label, audio
        index, and metadata.
        Args:
            index (int): the audio index provided by the pytorch sampler.
        Returns:
            spectrogram (tensor): the spectrogram sampled from the audio. The dimension
                is `channel` x `num frames` x `num frequencies`.
            label (int): the label of the current audio.
            index (int): Return the index of the audio.
        """


        if self.mode in ["train", "val", "train+val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
        elif self.mode in ["test"]:
            temporal_sample_index = self._temporal_idx[index]
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        spectrogram = pack_audio(self.cfg, self._audio_records[index], temporal_sample_index)
        # Normalization.
        spectrogram = spectrogram.float()
        if self.mode in ["train", "train+val"]:
            # Data augmentation.
            # C T F -> C F T
            spectrogram = spectrogram.permute(0, 2, 1)
            # SpecAugment
            if self.cfg.AUGMENTATION.AUG_METHOD == "double":
                spectrogram = combined_two_transforms(spectrogram)
            else:
                spectrogram = combined_transforms(spectrogram)
            # C F T -> C T F
            spectrogram = spectrogram.permute(0, 2, 1)
        label = self._audio_records[index]['class_id']
        spectrogram = utils.pack_pathway_output(self.cfg, spectrogram)

        return spectrogram, label, index, self._audio_records[index]['video']

    def __len__(self):
        return len(self._audio_records)
