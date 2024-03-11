export PYTHONPATH=/data1/steven/audio/AudioRepInceptionNeXt:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/run_net.py --cfg configs/EPIC-KITCHENS-416x128/AudioRepInceptionNeXt.yaml --init_method tcp://localhost:9997 \
NUM_GPUS 4 \
OUTPUT_DIR /data1/steven/audio/AudioRepInceptionNeXt/eval_epic_416x128/AudioRepInceptionNeXt/epoch_30 \
EPICKITCHENS.AUDIO_DATA_FILE /data_ssd/DATA/EPIC-Kitchens-100-hdf5/EPIC-KITCHENS-100_audio.hdf5 \
EPICKITCHENS.ANNOTATIONS_DIR /data1/steven/audio/AudioRepInceptionNeXt \
TRAIN.ENABLE False \
TEST.ENABLE True \
MODEL.MERGE_MODE True \
TEST.CHECKPOINT_FILE_PATH /data1/steven/audio/AudioRepInceptionNeXt/checkpoints_epic_416x128/AudioRepInceptionNeXt/checkpoints/checkpoint_epoch_00030.pyth

