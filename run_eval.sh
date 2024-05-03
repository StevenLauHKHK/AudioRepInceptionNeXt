export PYTHONPATH=/path/to/project/AudioRepInceptionNeXt:$PYTHONPATH

# MODEL.MERGE_MODE and MODEL.OUTPUT_DIR are optional argument
# MODEL.MERGE_MODE means conducting the reparameterization technique to run the eval (True means reparametrized model, False means the original training architecture)
# MODEL.OUTPUT_DIR is used to declare the saving directory of the reparametrized model

# Eval on VGG
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/run_net.py --cfg configs/VGG-Sound/AudioRepInceptionNeXt.yaml --init_method tcp://localhost:9998 \
NUM_GPUS num_gpus \
OUTPUT_DIR /path/to/eval_output_dir \
VGGSOUND.AUDIO_DATA_DIR /path/to/dataset \
VGGSOUND.ANNOTATIONS_DIR /path/to/annotations \
TRAIN.ENABLE False \
TEST.ENABLE True \
MODEL.MERGE_MODE True \
MODEL.OUTPUT_DIR /output/path/to/reparameterized_model \
TEST.CHECKPOINT_FILE_PATH /path/to/experiment_dir/checkpoints/checkpoint_best.pyth

# on EPIC-Kitchens
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/run_net.py --cfg configs/EPIC-KITCHENS-416x128/AudioRepInceptionNeXt.yaml --init_method tcp://localhost:9997 \
NUM_GPUS num_gpus \
OUTPUT_DIR /path/to/eval_output_dir \
EPICKITCHENS.AUDIO_DATA_FILE /path/to/dataset/EPIC-KITCHENS-100_audio.hdf5 \
EPICKITCHENS.ANNOTATIONS_DIR /path/to/annotations \
TRAIN.ENABLE False \
TEST.ENABLE True \
MODEL.MERGE_MODE True \
MODEL.OUTPUT_DIR /output/path/to/reparameterized_model \
TEST.CHECKPOINT_FILE_PATH /path/to/experiment_dir/checkpoints/checkpoint_best.pyth

# on EPIC-Sound
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/run_net.py --cfg configs/EPIC-SOUND-416x128/AudioRepInceptionNeXt.yaml --init_method tcp://localhost:9997 \
NUM_GPUS num_gpus \
OUTPUT_DIR /path/to/eval_output_dir \
EPICSOUND.AUDIO_DATA_FILE /path/to/dataset/EPIC-KITCHENS-100_audio.hdf5 \
EPICSOUND.ANNOTATIONS_DIR /path/to/annotations \
TRAIN.ENABLE False \
TEST.ENABLE True \
MODEL.MERGE_MODE True \
MODEL.OUTPUT_DIR /output/path/to/reparameterized_model \
TEST.CHECKPOINT_FILE_PATH /path/to/experiment_dir/checkpoints/checkpoint_best.pyth

# on Nsynth
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/run_net.py --cfg configs/Nsynth/AudioRepInceptionNeXt.yaml --init_method tcp://localhost:9997 \
NUM_GPUS num_gpus \
OUTPUT_DIR /path/to/eval_output_dir \
NSYNTH.VALID_AUDIO_DATA_DIR /path/to/dataset/nsynth-valid/audio \
NSYNTH.TEST_AUDIO_DATA_DIR /path/to/dataset/nsynth-valid/audio \
NSYNTH.TEST_LIST Nsynth-valid.pkl \
NSYNTH.ANNOTATIONS_DIR  /path/to/annotations \
TRAIN.ENABLE False \
TEST.ENABLE True \
MODEL.MERGE_MODE True \
MODEL.OUTPUT_DIR /output/path/to/reparameterized_model \
TEST.CHECKPOINT_FILE_PATH /path/to/experiment_dir/checkpoints/checkpoint_best.pyth

# on URBANSOUND8K
for i in {1..10};do
    
    CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/run_net.py --cfg configs/URBANSOUND8K/AudioRepInceptionNeXt.yaml --init_method tcp://localhost:9993 \
    NUM_GPUS num_gpus \
    OUTPUT_DIR /path/to/eval_output_dir/fold_$i \
    URBANSOUND8K.TEST_LIST UrbanSound8K_valid_10fold/UrbanSound8K_valid_fold$i.pkl \
    URBANSOUND8K.VAL_LIST UrbanSound8K_valid_10fold/UrbanSound8K_valid_fold$i.pkl \
    URBANSOUND8K.AUDIO_DATA_FILE /path/to/dataset/UrbanSound8K/audio \
    URBANSOUND8K.ANNOTATIONS_DIR /path/to/annotations \
    TRAIN.ENABLE False \
    TEST.ENABLE True \
    MODEL.MERGE_MODE True \
    MODEL.OUTPUT_DIR /output/path/to/reparameterized_model/fold_$i \
    TEST.CHECKPOINT_FILE_PATH /path/to/experiment_dir/fold_$i/checkpoints/checkpoint_epoch_00030.pyth
    
done

# on Speech-CMD-V2
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/run_net.py --cfg configs/SPEECH-CMD-V2/AudioRepInceptionNeXt.yaml --init_method tcp://localhost:9997 \
NUM_GPUS 4 \
OUTPUT_DIR /path/to/eval_output_dir \
SPEECHCOMMANDV2.TEST_LIST SPEECH_CMD_V2_validation.pkl \
SPEECHCOMMANDV2.AUDIO_DATA_FILE /path/to/dataset/Speech-Command-V2 \
SPEECHCOMMANDV2.ANNOTATIONS_DIR  /path/to/annotations \
TRAIN.ENABLE False \
TEST.ENABLE True \
MODEL.MERGE_MODE True \
MODEL.OUTPUT_DIR /output/path/to/reparameterized_model \
TEST.CHECKPOINT_FILE_PATH /path/to/experiment_dir/checkpoints/checkpoint_epoch_00030.pyth

