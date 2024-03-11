export PYTHONPATH=/data1/steven/audio/AudioRepInceptionNeXt:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/run_net.py --cfg configs/Nsynth/AudioRepInceptionNeXt.yaml --init_method tcp://localhost:9997 \
NUM_GPUS 4 \
OUTPUT_DIR /data1/steven/audio/AudioRepInceptionNeXt/eval_nsynth/valid/AudioRepInceptionNeXt/epoch_30 \
NSYNTH.VALID_AUDIO_DATA_DIR /data_ssd/DATA/nsynth_data/nsynth-valid/audio \
NSYNTH.TEST_AUDIO_DATA_DIR /data_ssd/DATA/nsynth_data/nsynth-valid/audio \
NSYNTH.TEST_LIST Nsynth-valid.pkl \
NSYNTH.ANNOTATIONS_DIR  /data1/steven/audio/AudioRepInceptionNeXt \
TRAIN.ENABLE False \
TEST.ENABLE True \
MODEL.MERGE_MODE True \
TEST.CHECKPOINT_FILE_PATH /data1/steven/audio/auditory-slow-fast/checkpoints_nsynth/FAST_R50_MS_K21_K11_K3_SE_DW_Mod1_Reduce_Resolution_All_Parallel_3_exp4/checkpoints/checkpoint_epoch_00030.pyth