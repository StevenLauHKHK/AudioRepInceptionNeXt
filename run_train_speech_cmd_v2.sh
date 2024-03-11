export PYTHONPATH=/data1/steven/audio/AudioRepInceptionNeXt:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/run_net.py --cfg configs/SPEECH-CMD-V2/AudioRepInceptionNeXt.yaml --init_method tcp://localhost:9996 \
NUM_GPUS 4 \
OUTPUT_DIR /data1/steven/audio/AudioRepInceptionNeXt/checkpoints_speech_cmd/AudioRepInceptionNeXt \
SPEECHCOMMANDV2.AUDIO_DATA_FILE /data_ssd/DATA/Speech-Command-V2 \
SPEECHCOMMANDV2.ANNOTATIONS_DIR  /data1/steven/audio/AudioRepInceptionNeXt \
TRAIN.CHECKPOINT_FILE_PATH /data1/steven/audio/AudioRepInceptionNeXt/checkpoints_vgg/AudioRepInceptionNeXt/checkpoints/checkpoint_epoch_00050.pyth
