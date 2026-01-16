"""
Orchestrator: Runs experiments sequentially.
Usage: python run_pipeline.py --experiment push_transfer
"""

import os
from calendar import c

import yaml

from scripts._01_train_dense import train_dense
from scripts._02_create_mask import create_mask
from scripts._03_train_ticket import train_mask


def main(train_agent=True, create_ticket=True, run_ticket=True):
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    tasks = cfg["environments"]["tasks"]
    seeds = cfg["environments"]["seeds"]

    for task in tasks:
        for seed in seeds:
            print(f"\n=== PROCESSING {task} SEED {seed} ===")

            ckpt_dir = f"data/checkpoints/{task}/seed_{seed}"
            mask_path = f"data/masks/{task}_seed_{seed}.pkl"

            os.makedirs(ckpt_dir, exist_ok=True)
            os.makedirs("data/masks", exist_ok=True)

            if train_agent:
                train_dense(cfg, task, seed, ckpt_dir)

            if create_ticket:
                create_mask(cfg, task, seed, ckpt_dir, mask_path)

            if run_ticket:
                ckpt_dir = ckpt_dir + "/ticket_training"
                os.makedirs(ckpt_dir, exist_ok=True)
                train_mask(cfg, task, seed, mask_path, ckpt_dir)


if __name__ == "__main__":
    main(train_agent=True, create_ticket=False, run_ticket=False)
