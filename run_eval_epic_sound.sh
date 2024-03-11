export PYTHONPATH=/data1/steven/audio/AudioRepInceptionNeXt:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/run_net.py --cfg configs/EPIC-SOUND-416x128/AudioRepInceptionNeXt.yaml --init_method tcp://localhost:9997 \
NUM_GPUS 4 \
OUTPUT_DIR /data1/steven/audio/AudioRepInceptionNeXt/eval_epic_sound_416x128/AudioRepInceptionNeXt/epoch_30 \
EPICSOUND.AUDIO_DATA_FILE /data_ssd/DATA/EPIC-Kitchens-100-hdf5/EPIC-KITCHENS-100_audio.hdf5 \
EPICSOUND.ANNOTATIONS_DIR /data1/steven/audio/AudioRepInceptionNeXt \
TRAIN.ENABLE False \
TEST.ENABLE True \
MODEL.MERGE_MODE True \
MODEL.OUTPUT_DIR /data1/steven/audio/AudioRepInceptionNeXt/inference_model_epic_sound/inference/AudioRepInceptionNeXt/best \
TEST.CHECKPOINT_FILE_PATH /data1/steven/audio/auditory-slow-fast/checkpoints_epic_sound_416x128/FAST_R50_MS_K21_K11_K3_SE_DW_Mod1_Reduce_Resolution_All_Parallel_3_exp4/checkpoints/checkpoint_epoch_00030.pyth
# TEST.CHECKPOINT_FILE_PATH /data1/steven/audio/AudioRepInceptionNeXt/checkpoints_epic_sounds/AudioRepInceptionNeXt/checkpoints/checkpoint_epoch_00030.pyth