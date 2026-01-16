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


def main(
    train_agent=True,
    create_ticket=True,
    run_ticket=True,
    pruning_method="gradient",  # "magnitude" or "gradient"
):
    """Run the lottery ticket hypothesis pipeline.
    
    Args:
        train_agent: Whether to train dense networks.
        create_ticket: Whether to create pruning masks.
        run_ticket: Whether to train lottery tickets.
        pruning_method: Pruning method - "magnitude" (faster) or "gradient" (better performance).
    """
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    tasks = cfg["environments"]["tasks"]
    seeds = cfg["environments"]["seeds"]
    
    # Validate pruning method
    if pruning_method not in ["magnitude", "gradient"]:
        raise ValueError(f"Invalid pruning_method '{pruning_method}'. Must be 'magnitude' or 'gradient'")
    
    print(f"\n{'='*80}")
    print(f"LOTTERY TICKET HYPOTHESIS PIPELINE")
    print(f"Pruning Method: {pruning_method.upper()}")
    print(f"{'='*80}\n")

    for task in tasks:
        for seed in seeds:
            print(f"\n=== PROCESSING {task} SEED {seed} ===")

            ckpt_dir = f"data/checkpoints/{task}/seed_{seed}"
            
            # Mask path includes pruning method
            mask_suffix = "" if pruning_method == "magnitude" else f"_{pruning_method}"
            mask_path = f"data/masks/{task}_seed_{seed}{mask_suffix}.pkl"

            os.makedirs(ckpt_dir, exist_ok=True)
            os.makedirs("data/masks", exist_ok=True)

            if train_agent:
                train_dense(cfg, task, seed, ckpt_dir)

            if create_ticket:
                create_mask(cfg, task, seed, ckpt_dir, mask_path, pruning_method=pruning_method)

            if run_ticket:
                ckpt_dir = ckpt_dir + "/ticket_training"
                os.makedirs(ckpt_dir, exist_ok=True)
                train_mask(cfg, task, seed, mask_path, ckpt_dir)


if __name__ == "__main__":
    # Configuration: Set what to run and which pruning method
    main(
        train_agent=True,
        create_ticket=True,
        run_ticket=True,
        pruning_method="gradient",  # Options: "magnitude" (faster) or "gradient" (better)
    )
