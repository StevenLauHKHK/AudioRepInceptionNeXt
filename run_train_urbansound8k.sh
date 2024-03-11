export PYTHONPATH=/data1/steven/audio/AudioRepInceptionNeXt:$PYTHONPATH


for i in {1..10};do

    CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/run_net.py --cfg configs/URBANSOUND8K/AudioRepInceptionNeXt.yaml --init_method tcp://localhost:9996 \
    NUM_GPUS 4 \
    OUTPUT_DIR /data1/steven/audio/AudioRepInceptionNeXt/checkpoints_urban_sound/fold_$i/AudioRepInceptionNeXt \
    URBANSOUND8K.AUDIO_DATA_FILE /data_ssd/DATA/UrbanSound8K/UrbanSound8K/audio \
    URBANSOUND8K.ANNOTATIONS_DIR  /data1/steven/audio/auditory-slow-fast \
    URBANSOUND8K.TRAIN_LIST UrbanSound8K_train_except_fold$i.pkl \
    URBANSOUND8K.VAL_LIST UrbanSound8K_valid_fold$i.pkl \
    URBANSOUND8K.TEST_LIST UrbanSound8K_valid_fold$i.pkl \
    TRAIN.CHECKPOINT_FILE_PATH /data1/steven/audio/AudioRepInceptionNeXt/checkpoints_vgg/AudioRepInceptionNeXt/checkpoints/checkpoint_epoch_00050.pyth

done