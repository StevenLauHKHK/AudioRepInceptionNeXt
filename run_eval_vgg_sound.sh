export PYTHONPATH=/data1/steven/audio/AudioRepInceptionNeXt:$PYTHONPATH

# MODEL.MERGE_MODE and MODEL.OUTPUT_DIR are optional argument
# MODEL.MERGE_MODE means conducting the reparameterization technique to run the eval (True means reparametrized model, False means the original training architecture)
# MODEL.OUTPUT_DIR is used to declare the saving directory of the reparametrized model

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/run_net.py --cfg configs/VGG-Sound/AudioRepInceptionNeXt.yaml --init_method tcp://localhost:9998 \
NUM_GPUS 4 \
OUTPUT_DIR /data1/steven/audio/AudioRepInceptionNeXt/eval/AudioRepInceptionNeXt/best \
VGGSOUND.AUDIO_DATA_DIR /data_ssd/DATA/VGGSound/wav_sound \
VGGSOUND.ANNOTATIONS_DIR /data1/steven/audio/AudioRepInceptionNeXt \
TRAIN.ENABLE False \
TEST.ENABLE True \
MODEL.MERGE_MODE True \
MODEL.OUTPUT_DIR /data1/steven/audio/AudioRepInceptionNeXt/inference_model_vgg_sound/inference/AudioRepInceptionNeXt/best \
TEST.CHECKPOINT_FILE_PATH /data1/steven/audio/AudioRepInceptionNeXt/checkpoints_vgg/AudioRepInceptionNeXt/checkpoints/checkpoint_best.pyth

