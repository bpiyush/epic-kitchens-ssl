"""Script to upload old runs to W&B via log file."""
from os.path import join, exists
from tqdm import tqdm
import wandb


def read_txt(path: str):
    with open(path, "rb") as f:
        lines = f.read()
        lines = lines.decode("utf-8")
        lines = lines.split("\n")
    return lines


def get_logged_stats(lines):
    stat_lines = []
    for line in lines:
        if "iter" in line and "epoch" in line:
            stat_lines.append(eval(line.split("json_stats: ")[-1]))
    
    return stat_lines


def init_wandb(name):
    wandb.init(project='epic-kitchens-ssl', entity='bpiyush', name=name)


if __name__ == "__main__":
    sample_path = "/home/pbagad/expts/epic-kitchens-ssl/R2PLUS1D_8x8_R2+1D_K400/logs/train_logs.txt"
    lines = read_txt(sample_path)
    stat_lines = get_logged_stats(lines)

    init_wandb(name="R2PLUS1D_8x8_R2+1D_K400")

    config = wandb.config
    config.config_name = "R2PLUS1D_8x8_R2+1D_K400.yaml"

    for line in tqdm(stat_lines, desc="Logging on W&B"):
        line.update(
            {"epoch": line["epoch"].split("/")[0]}
        )
        wandb.log(line)
