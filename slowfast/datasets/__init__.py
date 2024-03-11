#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .build import DATASET_REGISTRY, build_dataset  # noqa
from .vggsound import Vggsound
from .epicsound import Epicsound
from .speech_command_v2 import Speechcommandv2
from .urbansound8k import Urbansound8k
from .nsynth import Nsynth
from .epickitchens import Epickitchens

