export PYTHONPATH=/data1/steven/audio/AudioRepInceptionNeXt:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/run_net.py --cfg configs/Nsynth/AudioRepInceptionNeXt.yaml --init_method tcp://localhost:9996 \
NUM_GPUS 4 \
OUTPUT_DIR /data1/steven/audio/AudioRepInceptionNeXt/checkpoints_nsynth/AudioRepInceptionNeXt \
NSYNTH.TRAIN_AUDIO_DATA_DIR /data_ssd/DATA/nsynth_data/nsynth-train/audio \
NSYNTH.VALID_AUDIO_DATA_DIR /data_ssd/DATA/nsynth_data/nsynth-valid/audio \
NSYNTH.TEST_AUDIO_DATA_DIR /data_ssd/DATA/nsynth_data/nsynth-test/audio \
NSYNTH.ANNOTATIONS_DIR  /data1/steven/audio/AudioRepInceptionNeXt \
TRAIN.CHECKPOINT_FILE_PATH /data1/steven/audio/AudioRepInceptionNeXt/checkpoints_vgg/AudioRepInceptionNeXt/checkpoints/checkpoint_epoch_00050.pyth