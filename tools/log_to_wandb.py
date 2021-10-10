"""Script to upload old runs to W&B via log file."""
from os.path import join, exists, basename, splitext
from tqdm import tqdm
import wandb
import pandas as pd
import numpy as np

from tools.run_net import parse_args, load_config


def read_txt(path: str):
    with open(path, "rb") as f:
        lines = f.read()
        lines = lines.decode("utf-8")
        lines = lines.split("\n")
    return lines


def get_logged_stats(lines):
    stat_lines = []
    for line in lines:
        if "epoch" in line and "_type" in line:
            stat_lines.append(eval(line.split("[INFO: logging.py:   67]: json_stats: ")[-1]))

    return stat_lines


def init_wandb(name, project="video-ssl", entity="uva-vislab", dir="/var/scratch/pbagad"):
    wandb.init(name=name, project=project, entity=entity, dir=dir)


def get_iter_count(ep, it):
    it, num_samples = it.split("/")
    it = int(it)
    num_samples = int(num_samples)


if __name__ == "__main__":
    # log_path = "/home/pbagad/expts/epic-kitchens-ssl/R2PLUS1D_8x8_R2+1D_K400/logs/train_logs.txt"
    # log_path = "/home/pbagad/expts/epic-kitchens-ssl/SLOWFAST_8x8_R50_k400-pretrain/logs/train_logs.txt"
    # log_path = "/home/pbagad/expts/epic-kitchens-ssl/CTP_8x8_R2Plus1D_k400/logs/train_logs_1.txt"

    # SlowFast
    # log_path = "/var/scratch/pbagad/expts/epic-kitchens-ssl/SLOWFAST_8x8_R50_k400-pretrain/logs/train_logs.txt"
    # cfg_path = "/var/scratch/pbagad/projects/epic-kitchens-ssl/configs/EPIC-KITCHENS/SLOWFAST_8x8_R50_k400-pretrain.yaml"

    # CTP Model
    # log_path = "/var/scratch/pbagad/projects/epic-kitchens-slowfast/logs/ctp_on_epic_train.txt"
    # cfg_path = "/var/scratch/pbagad/projects/epic-kitchens-slowfast/configs/EPIC-KITCHENS/CTP_8x8_R2Plus1D_k400.yaml"

    # R2+1D with original hyperparameters
    # log_path = "/var/scratch/pbagad/projects/epic-kitchens-slowfast/logs/r2+1d_on_epic_train_from_k400-pretrain.txt"
    # cfg_path = "/var/scratch/pbagad/projects/epic-kitchens-slowfast/configs/EPIC-KITCHENS/R2PLUS1D_8x8_R2+1D_K400.yaml"

    # R2+1D with LR = 0.0025
    # log_path = "/home/pbagad/expts/epic-kitchens-ssl/32x112x112_R18_K400_LR0.0025/logs/train_logs.txt"
    # cfg_path = "/home/pbagad/projects/epic-kitchens-ssl/configs/EPIC-KITCHENS/R2PLUS1D/32x112x112_R18_K400_LR0.0025.yaml"
    
    # R2+1D with LR = 1e-4
    log_path = "/home/pbagad/expts/epic-kitchens-ssl/32x112x112_R18_K400_LR0.0001/logs/train_logs.txt"
    cfg_path = "/home/pbagad/projects/epic-kitchens-ssl/configs/EPIC-KITCHENS/R2PLUS1D/32x112x112_R18_K400_LR0.0025.yaml"

    # R2+1D with LR 0.0025 and consecutive 32 frames
    # log_path = "/var/scratch/pbagad/projects/epic-kitchens-slowfast/logs/consecutive_32x112x112_R18_K400_LR0.0025.txt"
    # cfg_path = "/var/scratch/pbagad/projects/epic-kitchens-slowfast/configs/EPIC-KITCHENS/R2PLUS1D/consecutive_32x112x112_R18_K400_LR0.0025.yaml"

    # load cfg
    args = parse_args()
    args.cfg_file = cfg_path
    cfg = load_config(args)

    lines = read_txt(log_path)
    stat_lines = get_logged_stats(lines)
    df = pd.DataFrame(stat_lines)

    # add column for epochs
    df["epoch"] = df["epoch"].apply(lambda x: x.split("/")[0]).astype(int)

    # add column for iterations
    train_total_iter = 0
    val_total_iter = 0
    for i in range(len(df)):
        iter = df.iloc[i]["iter"]
        _type = df.iloc[i]["_type"]
        if isinstance(iter, str):
            if "train" in _type:
                train_total_iter += int(iter.split("/")[0])
                df.at[i, "step"] = int(train_total_iter)
            if "val" in _type:
                val_total_iter += int(iter.split("/")[0])
                df.at[i, "step"] = int(val_total_iter)

    # df["step"] = df[["epoch", "iter"]].apply(lambda x: x[1]., axis=1)
    df = df.dropna(subset=["step"])
    df.step = df.step.astype(int)
    steps = df.step.values

    losses = [x for x in df.columns if "loss" in x]
    metrics = [x for x in df.columns if "_acc" in x]

    # initialize W&B
    init_wandb(name=splitext(basename(cfg_path))[0], entity="uva-vislab", project="video-ssl")

    # upload config
    config = wandb.config
    config.config_name = basename(cfg_path)
    config.update(cfg)

    # log train stats
    train_df = df[df["_type"].apply(lambda x: "train" in x)]
    train_df = train_df.rename(columns={k: "train/" + k for k in (metrics + losses + ["epoch", "step"])})
    train_records = train_df.to_dict('records')
    for r in tqdm(train_records, desc="Logging train stats"):
        wandb.log(r, commit=False)

    # log val stats
    val_df = df[df["_type"].apply(lambda x: "val" in x)]
    val_df = val_df.rename(columns={k: "val/" + k for k in (metrics + losses + ["epoch", "step"])})
    val_records = val_df.to_dict('records')
    for r in tqdm(val_records, desc="Logging val stats"):
        wandb.log(r)

