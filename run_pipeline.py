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
    use_iterative_pruning=False,  # Enable iterative pruning
):
    """Run the lottery ticket hypothesis pipeline.

    Args:
        train_agent: Whether to train dense networks.
        create_ticket: Whether to create pruning masks.
        run_ticket: Whether to train lottery tickets.
        pruning_method: Pruning method - "magnitude" (faster) or "gradient" (better performance).
        use_iterative_pruning: Whether to use iterative pruning instead of one-shot.
    """
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    tasks = cfg["environments"]["tasks"]
    seeds = cfg["environments"]["seeds"]

    # Validate pruning method
    if pruning_method not in ["magnitude", "gradient"]:
        raise ValueError(
            f"Invalid pruning_method '{pruning_method}'. Must be 'magnitude' or 'gradient'"
        )

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
                if use_iterative_pruning:
                    from scripts.iterative_pruning import iterative_pruning
                    print(f"\n  >> Using ITERATIVE PRUNING <<")
                    iterative_pruning(
                        cfg, task, seed, ckpt_dir, mask_path, pruning_method=pruning_method
                    )
                else:
                    create_mask(
                        cfg, task, seed, ckpt_dir, mask_path, pruning_method=pruning_method
                    )

            if run_ticket:
                base_ticket_dir = ckpt_dir + "/ticket_training/gradient" if pruning_method == "gradient" else ckpt_dir + "/ticket_training/magnitude"
                
                if use_iterative_pruning:
                    # Train tickets for each iteration
                    iterative_cfg = cfg.get("pruning", {}).get("iterative", {})
                    num_iterations = iterative_cfg.get("num_iterations", 4)
                    
                    for iteration in range(1, num_iterations + 1):
                        iteration_dir = os.path.join(base_ticket_dir, f"iteration_{iteration:02d}")
                        os.makedirs(iteration_dir, exist_ok=True)
                        
                        # Load iteration-specific mask
                        iter_mask_dir = os.path.join(os.path.dirname(mask_path), "iterations")
                        iter_mask_path = os.path.join(iter_mask_dir, f"mask_iter_{iteration}_{pruning_method}.pkl")
                        
                        if os.path.exists(iter_mask_path):
                            print(f"\n  >> Training Ticket for Iteration {iteration}/{num_iterations}")
                            train_mask(cfg, task, seed, iter_mask_path, iteration_dir)
                        else:
                            print(f"  [Skip] Mask not found for iteration {iteration}: {iter_mask_path}")
                else:
                    # Single ticket training (non-iterative)
                    os.makedirs(base_ticket_dir, exist_ok=True)
                    train_mask(cfg, task, seed, mask_path, base_ticket_dir)


if __name__ == "__main__":
    # Configuration: Set what to run and which pruning method
    main(
        train_agent=True,
        create_ticket=True,
        run_ticket=True,
        pruning_method="gradient",  # Options: "magnitude" (faster) or "gradient" (better)
        use_iterative_pruning=True,  # Set to True for iterative pruning
    )
