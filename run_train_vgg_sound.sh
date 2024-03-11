export PYTHONPATH=/data1/steven/audio/AudioRepInceptionNeXt:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/run_net.py --cfg configs/VGG-Sound/AudioRepInceptionNeXt.yaml --init_method tcp://localhost:9996 \
NUM_GPUS 4 \
OUTPUT_DIR /data1/steven/audio/AudioRepInceptionNeXt/checkpoints_vgg/AudioRepInceptionNeXt \
VGGSOUND.AUDIO_DATA_DIR /data_ssd/DATA/VGGSound/wav_sound \
VGGSOUND.ANNOTATIONS_DIR /data1/steven/audio/AudioRepInceptionNeXt




