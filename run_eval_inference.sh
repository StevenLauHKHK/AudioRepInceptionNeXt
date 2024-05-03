export PYTHONPATH=/path/to/project/AudioRepInceptionNeXt:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/run_net.py --cfg configs/VGG-Sound/AudioRepInceptionNeXt_Inference.yaml --init_method tcp://localhost:9998 \
NUM_GPUS num_gpus \
OUTPUT_DIR /path/to/eval_output_dir \
VGGSOUND.AUDIO_DATA_DIR /path/to/dataset \
VGGSOUND.ANNOTATIONS_DIR /path/to/annotations \
TRAIN.ENABLE False \
TEST.ENABLE True \
TEST.CHECKPOINT_FILE_PATH /path/to/new_model_saving_dir/checkpoints/checkpoint_best.pyth
