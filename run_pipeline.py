"""
Orchestrator: Runs Iterative Pruning (IMP) with Late Rewinding.
Usage: python run_pipeline.py
"""

import gc
import math
import os
import shutil

import jax
import yaml

from scripts._01_train_dense import train_dense
from scripts._02_create_mask import create_mask
from scripts._03_train_ticket import train_mask


def clean_memory():
    """Forces garbage collection and clears JAX caches."""
    gc.collect()  # Force Python to release unreferenced objects (like ReplayBuffers)
    jax.clear_caches()  # Clear JAX compilation caches (helps prevent recompilation bloat)
    # Note: JAX doesn't have a simple 'reset_gpu_memory' like PyTorch,
    # but deleting variables + gc.collect() usually works.


def get_dual_schedule(target_actor, target_critic, rate):
    """
    Generates a schedule where Actor goes to target_actor,
    but Critic stops early or stays dense.
    """
    schedule = []
    curr_actor = 0.0
    curr_critic = 0.0

    # Continue until Actor reaches target
    while curr_actor < target_actor:
        # Update Actor
        next_actor = curr_actor + (1 - curr_actor) * rate
        if next_actor > target_actor:
            next_actor = target_actor

        # Update Critic (Only if target > 0)
        if target_critic > 0:
            next_critic = curr_critic + (1 - curr_critic) * rate
            if next_critic > target_critic:
                next_critic = target_critic
        else:
            next_critic = 0.0

        schedule.append((next_actor, next_critic))
        curr_actor = next_actor
        curr_critic = next_critic

        # Break if we're basically there (floating point safety)
        if curr_actor >= target_actor and curr_critic >= target_critic:
            break

    return schedule


def main(
    train_agent=True,
    create_ticket=True,
    run_ticket=True,
    pruning_method="magnitude",
    target_sparsity_actor=0.80,  # 80% Sparse Actor
    target_sparsity_critic=0.00,  # 0% Sparse Critic (Dense)
    pruning_rate=0.33,
):
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    tasks = cfg["environments"]["tasks"]
    seeds = cfg["environments"]["seeds"]

    # Calculate how many steps to rewind to (e.g., 5% of total steps)
    total_steps = cfg["hyperparameters"]["total_steps"]
    rewind_steps = cfg.get("pruning", {}).get("rewind_steps", 0)

    print(f"\n{'='*80}")
    print(f"ITERATIVE LOTTERY TICKET PIPELINE ({pruning_method.upper()})")
    print(
        f"Targets >> Actor: {target_sparsity_actor:.1%} | Critic: {target_sparsity_critic:.1%}"
    )
    print(f"Pruning Rate: {pruning_rate:.0%}")
    print(f"Late Rewinding to Step: {rewind_steps}")
    print(f"{'='*80}\n")

    for task in tasks:
        for seed in seeds:
            print(f"\n>>> PROCESSING {task} | SEED {seed} <<<")

            base_exp_dir = f"data/experiments/{task}/seed_{seed}"
            os.makedirs(base_exp_dir, exist_ok=True)

            # --- ROUND 0: Dense Training & Rewind Anchor ---
            print(f"\n[Round 0] Initial Dense Training")
            round_0_dir = os.path.join(base_exp_dir, "round_0")
            os.makedirs(round_0_dir, exist_ok=True)

            rewind_ckpt_path = os.path.join(round_0_dir, "checkpoint_rewind.pkl")

            if train_agent:
                train_dense(
                    cfg,
                    task,
                    seed,
                    save_dir=round_0_dir,
                    rewind_steps=rewind_steps,
                )
                clean_memory()
            else:
                print("  [Skip] Dense training disabled via flag.")

            # --- ITERATIVE PRUNING LOOP ---
            schedule = get_dual_schedule(
                target_sparsity_actor, target_sparsity_critic, pruning_rate
            )

            prev_round_dir = round_0_dir
            prev_mask_path = None

            # Unpack tuple here: (act_sp, crit_sp)
            for i, (act_sp, crit_sp) in enumerate(schedule):
                round_num = i + 1
                print(
                    f"\n[Round {round_num}] Targets >> Actor: {act_sp:.2%} | Critic: {crit_sp:.2%}"
                )

                current_round_dir = os.path.join(base_exp_dir, f"round_{round_num}")
                os.makedirs(current_round_dir, exist_ok=True)

                mask_path = os.path.join(current_round_dir, "mask.pkl")

                # 1. Create Mask (Pass separate targets!)
                if create_ticket:
                    create_mask(
                        cfg,
                        task,
                        seed,
                        ckpt_dir=prev_round_dir,
                        mask_out_path=mask_path,
                        # Pass explicit separate targets
                        target_sparsity_actor=act_sp,
                        target_sparsity_critic=crit_sp,
                        pruning_method=pruning_method,
                        prev_mask_path=prev_mask_path,
                    )
                    clean_memory()
                else:
                    print("  [Skip] Mask creation disabled via flag.")

                # 2. Train Ticket
                if run_ticket:
                    if not os.path.exists(mask_path):
                        print(f"  [Warning] Mask not found at {mask_path}")
                    else:
                        train_mask(
                            cfg,
                            task,
                            seed,
                            mask_path=mask_path,
                            save_dir=current_round_dir,
                            rewind_ckpt_path=rewind_ckpt_path,
                        )
                        clean_memory()
                else:
                    print("  [Skip] Ticket training disabled via flag.")

                prev_round_dir = current_round_dir
                prev_mask_path = mask_path


if __name__ == "__main__":
    main(
        train_agent=True,
        create_ticket=True,
        run_ticket=True,
        pruning_method="gradient",
        target_sparsity_actor=0.80,
        target_sparsity_critic=0.00,
        pruning_rate=0.33,
    )
