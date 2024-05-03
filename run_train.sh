export PYTHONPATH=/path/to/project/AudioRepInceptionNeXt:$PYTHONPATH

# Pretraining on VGG
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/run_net.py --cfg configs/VGG-Sound/AudioRepInceptionNeXt.yaml --init_method tcp://localhost:9996 \
NUM_GPUS num_gpus \
OUTPUT_DIR /path/to/output_dir \
VGGSOUND.AUDIO_DATA_DIR /path/to/dataset  \
VGGSOUND.ANNOTATIONS_DIR /path/to/annotations 

# Fine tuning on EPIC-Kitchens
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/run_net.py --cfg configs/EPIC-KITCHENS-416x128/AudioRepInceptionNeXt.yaml --init_method tcp://localhost:9996 \
NUM_GPUS num_gpus \
OUTPUT_DIR /path/to/output_dir \
EPICKITCHENS.AUDIO_DATA_FILE /path/to/dataset/EPIC-KITCHENS-100_audio.hdf5 \
EPICKITCHENS.ANNOTATIONS_DIR  /path/to/annotations \
TRAIN.CHECKPOINT_FILE_PATH /path/to/pretraining/experiment_dir/checkpoints/checkpoint_epoch_00050.pyth

# Fine tuning on EPIC-Sound
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/run_net.py --cfg configs/EPIC-SOUND-416x128/AudioRepInceptionNeXt.yaml --init_method tcp://localhost:9996 \
NUM_GPUS num_gpus \
OUTPUT_DIR /path/to/output_dir \
EPICSOUND.AUDIO_DATA_FILE /path/to/dataset/EPIC-KITCHENS-100_audio.hdf5 \
EPICSOUND.ANNOTATIONS_DIR  /path/to/annotations \
TRAIN.CHECKPOINT_FILE_PATH /path/to/pretraining/experiment_dir/checkpoints/checkpoint_epoch_00050.pyth

# Fine tuning on Nsynth
CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/run_net.py --cfg configs/Nsynth/AudioRepInceptionNeXt.yaml --init_method tcp://localhost:9996 \
NUM_GPUS num_gpus \
OUTPUT_DIR /path/to/output_dir \
NSYNTH.TRAIN_AUDIO_DATA_DIR /path/to/dataset/nsynth-train/audio \
NSYNTH.VALID_AUDIO_DATA_DIR /path/to/dataset/nsynth-valid/audio \
NSYNTH.TEST_AUDIO_DATA_DIR /path/to/dataset/nsynth-test/audio \
NSYNTH.ANNOTATIONS_DIR  /path/to/annotations \
TRAIN.CHECKPOINT_FILE_PATH /path/to/pretraining/experiment_dir/checkpoints/checkpoint_epoch_00050.pyth

# Fine tuning on URBANSOUND8K
for i in {1..10};do

    CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/run_net.py --cfg configs/URBANSOUND8K/AudioRepInceptionNeXt.yaml --init_method tcp://localhost:9996 \
    NUM_GPUS num_gpus \
    OUTPUT_DIR /path/to/output_dir/fold_$i/AudioRepInceptionNeXt \
    URBANSOUND8K.AUDIO_DATA_FILE /path/to/dataset/UrbanSound8K/audio \
    URBANSOUND8K.ANNOTATIONS_DIR  /path/to/annotations \
    URBANSOUND8K.TRAIN_LIST UrbanSound8K_train_except_fold$i.pkl \
    URBANSOUND8K.VAL_LIST UrbanSound8K_valid_fold$i.pkl \
    URBANSOUND8K.TEST_LIST UrbanSound8K_valid_fold$i.pkl \
    TRAIN.CHECKPOINT_FILE_PATH /path/to/pretraining/experiment_dir/checkpoints/checkpoint_epoch_00050.pyth

done

# Fine tuning on Speech-CMD-V2

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/run_net.py --cfg configs/SPEECH-CMD-V2/AudioRepInceptionNeXt.yaml --init_method tcp://localhost:9996 \
NUM_GPUS num_gpus \
OUTPUT_DIR /path/to/output_dir \
SPEECHCOMMANDV2.AUDIO_DATA_FILE /path/to/dataset/Speech-Command-V2 \
SPEECHCOMMANDV2.ANNOTATIONS_DIR  /path/to/annotations \
TRAIN.CHECKPOINT_FILE_PATH /path/to/pretraining/experiment_dir/checkpoints/checkpoint_epoch_00050.pyth

