# helper script to run sample training run
repo="$(cd "$(dirname "$1")"; pwd -P)/$(basename "$1")"
export PYTHONPATH=$repo

# expt details
# cfg=configs/EPIC-KITCHENS/SLOWFAST_8x8_R50_k400-pretrain.yaml
cfg=configs/EPIC-KITCHENS/R2PLUS1D/32x112x112_R18_K400_LR0.0025.yaml
num_gpus=4
# train_ckpt_path=/home/pbagad/expts/epic-kitchens-ssl/pretrained/SLOWFAST_8x8_R50.pkl

# output paths
expt_folder="$(basename -- $cfg)"
expt_folder="${expt_folder%.yaml}"
output_dir=/home/pbagad/expts/epic-kitchens-ssl/$expt_folder/
echo "Saving outputs: "$output_dir
mkdir -p $output_dir
logs_dir=$output_dir/logs/
mkdir -p $logs_dir

# dataset paths
dataset_dir=/ssd/pbagad/datasets/EPIC-KITCHENS-100/EPIC-KITCHENS/
annotations_dir=/ssd/pbagad/datasets/EPIC-KITCHENS-100/annotations/

# run training
python tools/run_net.py \
    --cfg $cfg \
    NUM_GPUS $num_gpus \
    OUTPUT_DIR $output_dir \
    EPICKITCHENS.VISUAL_DATA_DIR $dataset_dir \
    EPICKITCHENS.ANNOTATIONS_DIR $annotations_dir > $logs_dir/train_logs.txt \
    # TRAIN.CHECKPOINT_FILE_PATH $train_ckpt_path
