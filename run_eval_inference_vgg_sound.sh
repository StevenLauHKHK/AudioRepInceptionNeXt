export PYTHONPATH=/data1/steven/audio/AudioRepInceptionNeXt:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/run_net.py --cfg configs/VGG-Sound/AudioRepInceptionNeXt_Inference.yaml --init_method tcp://localhost:9998 \
NUM_GPUS 4 \
OUTPUT_DIR /data1/steven/audio/AudioRepInceptionNeXt/eval/AudioRepInceptionNeXt_Inference/best \
VGGSOUND.AUDIO_DATA_DIR /data_ssd/DATA/VGGSound/wav_sound \
VGGSOUND.ANNOTATIONS_DIR /data1/steven/audio/AudioRepInceptionNeXt \
TRAIN.ENABLE False \
TEST.ENABLE True \
TEST.CHECKPOINT_FILE_PATH /data1/steven/audio/AudioRepInceptionNeXt/inference/AudioRepInceptionNeXt/best/checkpoints/checkpoint_epoch_00048.pyth
