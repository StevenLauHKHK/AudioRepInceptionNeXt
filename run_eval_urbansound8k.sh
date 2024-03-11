export PYTHONPATH=/data1/steven/audio/AudioRepInceptionNeXt:$PYTHONPATH

for i in {1..10};do
    
    CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/run_net.py --cfg configs/URBANSOUND8K/AudioRepInceptionNeXt.yaml --init_method tcp://localhost:9993 \
    NUM_GPUS 4 \
    OUTPUT_DIR /data1/steven/audio/AudioRepInceptionNeXt/eval_valid_urbansound8k/fold_$i/AudioRepInceptionNeXt/epoch_30 \
    URBANSOUND8K.TEST_LIST UrbanSound8K_valid_10fold/UrbanSound8K_valid_fold$i.pkl \
    URBANSOUND8K.VAL_LIST UrbanSound8K_valid_10fold/UrbanSound8K_valid_fold$i.pkl \
    URBANSOUND8K.AUDIO_DATA_FILE /data_ssd/DATA/UrbanSound8K/UrbanSound8K/audio \
    URBANSOUND8K.ANNOTATIONS_DIR  /data1/steven/audio/AudioRepInceptionNeXt \
    TRAIN.ENABLE False \
    TEST.ENABLE True \
    MODEL.MERGE_MODE True \
    TEST.CHECKPOINT_FILE_PATH /data1/steven/audio/auditory-slow-fast/checkpoints_urban_sound/fold_$i/FAST_R50_MS_K21_K11_K3_SE_DW_Mod1_Reduce_Resolution_All_Parallel_3_exp4/checkpoints/checkpoint_epoch_00030.pyth

done