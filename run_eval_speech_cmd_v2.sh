export PYTHONPATH=/data1/steven/audio/AudioRepInceptionNeXt:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/run_net.py --cfg configs/SPEECH-CMD-V2/AudioRepInceptionNeXt.yaml --init_method tcp://localhost:9997 \
NUM_GPUS 4 \
OUTPUT_DIR /data1/steven/audio/AudioRepInceptionNeXt/eval_valid_speech_cmd/AudioRepInceptionNeXt/epoch_30 \
SPEECHCOMMANDV2.TEST_LIST SPEECH_CMD_V2_validation.pkl \
SPEECHCOMMANDV2.AUDIO_DATA_FILE /data_ssd/DATA/Speech-Command-V2 \
SPEECHCOMMANDV2.ANNOTATIONS_DIR  /data1/steven/audio/AudioRepInceptionNeXt \
TRAIN.ENABLE False \
TEST.ENABLE True \
MODEL.MERGE_MODE True \
TEST.CHECKPOINT_FILE_PATH /data1/steven/audio/auditory-slow-fast/checkpoints_speech_cmd/FAST_R50_MS_K21_K11_K3_SE_DW_Mod1_Reduce_Resolution_All_Parallel_3_exp6/checkpoints/checkpoint_epoch_00030.pyth