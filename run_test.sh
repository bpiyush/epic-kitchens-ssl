# helper script to run sample training run
repo="$(cd "$(dirname "$1")"; pwd -P)/$(basename "$1")"
export PYTHONPATH=$repo

# expt details
cfg=configs/EPIC-KITCHENS/SLOWFAST_8x8_R50_k400-pretrain.yaml
num_gpus=1

# output paths
expt_folder="$(basename -- $cfg)"
expt_folder="${expt_folder%.yaml}"
output_dir=/home/pbagad/expts/epic-kitchens-ssl/$expt_folder/
echo "Saving outputs: "$output_dir
mkdir -p $output_dir
logs_dir=$output_dir/logs/
mkdir -p $logs_dir

# checkpoint path
# ckpt_path=/home/pbagad/expts/epic-kitchens-ssl/pretrained/SlowFast.pyth
ckpt_path=/home/pbagad/expts/epic-kitchens-ssl/SLOWFAST_8x8_R50_k400-pretrain/checkpoints/checkpoint_epoch_00030.pyth

# dataset paths
dataset_dir=/home/pbagad/datasets/EPIC-KITCHENS-100/EPIC-KITCHENS/
annotations_dir=/home/pbagad/datasets/EPIC-KITCHENS-100/annotations/

# run training
python tools/run_net.py \
    --cfg $cfg \
    NUM_GPUS $num_gpus \
    OUTPUT_DIR $output_dir \
    EPICKITCHENS.VISUAL_DATA_DIR $dataset_dir \
    EPICKITCHENS.ANNOTATIONS_DIR $annotations_dir \
    TEST.CHECKPOINT_FILE_PATH $ckpt_path \
    TRAIN.ENABLE False \
    TEST.ENABLE True \

# > $logs_dir/val_logs.txt \
