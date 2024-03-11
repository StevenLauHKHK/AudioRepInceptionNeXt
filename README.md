# Auditory AudioRepInceptionNeXt

This repository implements the model proposed in the paper:

Kin Wai Lau, Yasar Abbas Ur Rehman, Lai-Man Po, **AudioRepInceptionNeXt: A lightweight single-stream architecture for efficient audio recognition**

[[arXiv paper]](Coming Soon)

The implementation code is based on the **Slow-Fast Auditory Streams for Audio Recognition**, ICASSP, 2021. For more information, please refer to the [link](https://github.com/ekazakos/auditory-slow-fast).


## Citing

When using this code, kindly reference:

```
@article{lau2024audiorepinceptionnext,
  title={AudioRepInceptionNeXt: A lightweight single-stream architecture for efficient audio recognition},
  author={Lau, Kin Wai and Rehman, Yasar Abbas Ur and Po, Lai-Man},
  journal={Neurocomputing},
  pages={127432},
  year={2024},
  publisher={Elsevier}
}
```

## Pretrained models

You can download our pretrained models as follow:
- AudioRepInceptionNeXt (VGG-Sound) [link](https://portland-my.sharepoint.com/:f:/g/personal/kinwailau6-c_my_cityu_edu_hk/EiYfDsGXvLRNsGJEJ8EuNIIBm3BaWXQsmFAmRP7ZEucbuw?e=YUbTEM)
- AudioRepInceptionNeXt (EPIC-Sound) [link](https://portland-my.sharepoint.com/:f:/g/personal/kinwailau6-c_my_cityu_edu_hk/Evws-ER1bFRHnrfADk0awVgBBKskaFgokCAK52cuzJNbwQ?e=Pvufdm)
- AudioRepInceptionNeXt (EPIC-Kitchens-100) [link] (https://portland-my.sharepoint.com/:f:/g/personal/kinwailau6-c_my_cityu_edu_hk/El4P4d8wSYhOibuNdauOdY0BY6tbsMJjEmxxZZ4EvuxZ9A?e=v7F0TZ)
- AudioRepInceptionNeXt (Speech Commands V2) [link] (https://portland-my.sharepoint.com/:f:/g/personal/kinwailau6-c_my_cityu_edu_hk/Eg6t5eGNPtdIrCGQo-o1xdABiJ-HJHd6Yx9yUhvQIEkw1Q?e=I8OVwd)
- AudioRepInceptionNeXt (Urban Sound 8K) [link] (https://portland-my.sharepoint.com/:f:/g/personal/kinwailau6-c_my_cityu_edu_hk/EpeRXtGnVoxAigfyrMSBN6ABCT0y1l5bsNQTmFaJVoZXtA?e=A7gPYt)
- AudioRepInceptionNeXt (NSynth) [link] (https://portland-my.sharepoint.com/:f:/g/personal/kinwailau6-c_my_cityu_edu_hk/Er4eZSf52DhOqcUYSWgd4sIBDEdAdLJxPz0g7gUwPpAwAw?e=v8Xo6e)



## Preparation

* Requirements:
  * [PyTorch](https://pytorch.org) 1.7.1
  * [librosa](https://librosa.org): `conda install -c conda-forge librosa`
  * [h5py](https://www.h5py.org): `conda install h5py`
  * [wandb](https://wandb.ai/site): `pip install wandb`
  * [fvcore](https://github.com/facebookresearch/fvcore/): `pip install 'git+https://github.com/facebookresearch/fvcore'`
  * simplejson: `pip install simplejson`
  * psutil: `pip install psutil`
  * tensorboard: `pip install tensorboard` 
* Add this repository to $PYTHONPATH.
```
export PYTHONPATH=/path/to/AudioRepInceptionNeXt:$PYTHONPATH
```
* VGG-Sound:
  See the instruction in Auditory Slow-Fast repository [link](https://github.com/ekazakos/auditory-slow-fast)
* EPIC-KITCHENS:
  See the instruction in Auditory Slow-Fast repository [link](https://github.com/ekazakos/auditory-slow-fast)
* EPIC-Sounds
  See the instruction in Epic-Sounds annotations repository [link](https://github.com/epic-kitchens/epic-sounds-annotations) and [link](https://github.com/epic-kitchens/epic-sounds-annotations/tree/main/src)

## Training/validation data
* VGG-Sound:
  URL of the dataset [link] (https://www.robots.ox.ac.uk/~vgg/data/vggsound/)

* EPIC-KITCHENS:
  URL of the dataset [link] (https://epic-kitchens.github.io/2024)

* EPIC-Sounds:
  URL of the dataset [link] (https://epic-kitchens.github.io/epic-sounds/)

* Speech Commands V2:
  URL of the dataset [link] (http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz)

* Urban Sound 8K:
  URL of the dataset [link] (https://urbansounddataset.weebly.com/urbansound8k.html)

* NSynth:
  URL of the dataset [link] (https://magenta.tensorflow.org/datasets/nsynth#files)


## Training/validation on VGG-Sound
To train the model run (see run_train_vgg_sound.sh as an example):
```
python tools/run_net.py --cfg configs/VGG-Sound/AudioRepInceptionNeXt.yaml --init_method tcp://localhost:9996 \
NUM_GPUS num_gpus \
OUTPUT_DIR /path/to/output_dir \
VGGSOUND.AUDIO_DATA_DIR /path/to/dataset 
VGGSOUND.ANNOTATIONS_DIR /path/to/annotations 
```

To validate the trained model run (see run_eval_vgg_sound.sh as an example):
```
python tools/run_net.py --cfg configs/VGG-Sound/AudioRepInceptionNeXt.yaml --init_method tcp://localhost:9998 \
NUM_GPUS num_gpus \
OUTPUT_DIR /path/to/experiment_dir \
VGGSOUND.AUDIO_DATA_DIR /path/to/dataset \
VGGSOUND.ANNOTATIONS_DIR /path/to/annotations \
TRAIN.ENABLE False \
TEST.ENABLE True \
TEST.CHECKPOINT_FILE_PATH /path/to/experiment_dir/checkpoints/checkpoint_best.pyth
```

To export the reparametrized AudioRepInceptionNeXt run (see run_eval_vgg_sound.sh as an example):
```
python tools/run_net.py --cfg configs/VGG-Sound/AudioRepInceptionNeXt.yaml --init_method tcp://localhost:9998 \
NUM_GPUS num_gpus \
OUTPUT_DIR /path/to/experiment_dir \
VGGSOUND.AUDIO_DATA_DIR /path/to/dataset \
VGGSOUND.ANNOTATIONS_DIR /path/to/annotations \
TRAIN.ENABLE False \
TEST.ENABLE True \
MODEL.MERGE_MODE True \
MODEL.OUTPUT_DIR /path/to/new_model_saving_dir \
TEST.CHECKPOINT_FILE_PATH /path/to/experiment_dir/checkpoints/checkpoint_best.pyth
```

To run the reparametrized AudioRepInceptionNeXt in inference mode run (see run_eval_inference_vgg_sound.sh as an example):
```
python tools/run_net.py --cfg configs/VGG-Sound/AudioRepInceptionNeXt_Inference.yaml --init_method tcp://localhost:9998 \
NUM_GPUS num_gpus \
OUTPUT_DIR /path/to/experiment_dir \
VGGSOUND.AUDIO_DATA_DIR /path/to/dataset \
VGGSOUND.ANNOTATIONS_DIR /path/to/annotations \
TRAIN.ENABLE False \
TEST.ENABLE True \
TEST.CHECKPOINT_FILE_PATH /path/to/new_model_saving_dir/checkpoints/checkpoint_best.pyth
```

## Fine Tune/validation on EPIC-Sounds
To fine-tuning from VGG-Sound pretrained model (see run_train_epic_sound.sh as an example):
```
python tools/run_net.py --cfg configs/EPIC-SOUND-416x128/AudioRepInceptionNeXt.yaml --init_method tcp://localhost:9996 \
NUM_GPUS num_gpus \
OUTPUT_DIR /path/to/output_dir \
EPICSOUND.AUDIO_DATA_FILE /path/to/EPIC-KITCHENS-100_audio.hdf5 \
EPICSOUND.ANNOTATIONS_DIR /path/to/annotations \
TRAIN.CHECKPOINT_FILE_PATH /path/to/VGG-Sound/pretrained/model
```

To validate the model run (see run_eval_epic_sound.sh as an example)::
```
python tools/run_net.py --cfg configs/EPIC-SOUND-416x128/AudioRepInceptionNeXt.yaml --init_method tcp://localhost:9997 \
NUM_GPUS num_gpus \
OUTPUT_DIR /path/to/experiment_dir \
EPICKITCHENS.AUDIO_DATA_FILE /path/to/EPIC-KITCHENS-100_audio.hdf5 \
EPICKITCHENS.ANNOTATIONS_DIR /path/to/annotations \
TRAIN.ENABLE False \
TEST.ENABLE True \
TEST.CHECKPOINT_FILE_PATH /path/to/experiment_dir/checkpoints/checkpoint_best.pyth
```



