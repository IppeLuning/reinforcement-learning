from __future__ import annotations

import gc
import os
from typing import Any, Dict, List, Optional, Tuple

import jax
import yaml

from scripts._01_train_dense import train_dense
from scripts._02_create_mask import create_mask
from scripts._03_train_ticket import train_mask


def clean_memory() -> None:
    """Forces garbage collection and clears JAX compilation caches.

    This is critical during iterative pruning to prevent Out-Of-Memory (OOM)
    errors when handling multiple ReplayBuffers and model states in a single process.
    """
    gc.collect()
    jax.clear_caches()


def get_dual_schedule(
    target_actor: float, target_critic: float, rate: float
) -> List[Tuple[float, float]]:
    """Generates an iterative pruning schedule for both Actor and Critic.

    The schedule follows the formula: $S_{n+1} = S_n + (1 - S_n) \times rate$,
    ensuring that sparsity targets are approached asymptotically until the
    defined targets are met.

    Args:
        target_actor: Final sparsity target for the Actor (0.0 to 1.0).
        target_critic: Final sparsity target for the Critic (0.0 to 1.0).
        rate: The fraction of remaining weights to prune in each round.

    Returns:
        A list of (actor_sparsity, critic_sparsity) tuples for each round.
    """
    schedule: List[Tuple[float, float]] = []
    curr_actor: float = 0.0
    curr_critic: float = 0.0

    while curr_actor < target_actor or curr_critic < target_critic:
        next_actor = curr_actor + (1 - curr_actor) * rate
        if next_actor > target_actor:
            next_actor = target_actor

        next_critic = curr_critic + (1 - curr_critic) * rate
        if next_critic > target_critic:
            next_critic = target_critic

        schedule.append((next_actor, next_critic))
        curr_actor = next_actor
        curr_critic = next_critic

        if curr_actor >= target_actor and curr_critic >= target_critic:
            break

    return schedule


def main(
    train_agent: bool = True,
    create_ticket: bool = True,
    run_ticket: bool = True,
    pruning_method: str = "magnitude",
    target_sparsity_actor: float = 0.80,
    target_sparsity_critic: float = 0.00,
    pruning_rate: float = 0.33,
) -> None:
    """Orchestrates the iterative pruning pipeline.

    Args:
        train_agent: Whether to run the initial Round 0 dense training.
        create_ticket: Whether to generate new masks at each round.
        run_ticket: Whether to perform the retraining phase for the sparse tickets.
        pruning_method: Criterion for pruning ("magnitude" or "gradient").
        target_sparsity_actor: Final desired sparsity for the Actor.
        target_sparsity_critic: Final desired sparsity for the Critic.
        pruning_rate: The rate of iterative pruning per round.
    """
    with open("config.yaml", "r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    tasks: List[str] = cfg["environments"]["tasks"]
    seeds: List[int] = cfg["environments"]["seeds"]
    rewind_steps: int = cfg.get("pruning", {}).get("rewind_steps", 0)

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

            base_exp_dir: str = f"data/experiments/{task}/seed_{seed}"
            os.makedirs(base_exp_dir, exist_ok=True)

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

            schedule = get_dual_schedule(
                target_sparsity_actor, target_sparsity_critic, pruning_rate
            )

            prev_round_dir = round_0_dir
            prev_mask_path: Optional[str] = None

            for i, (act_sp, crit_sp) in enumerate(schedule):
                round_num = i + 1
                print(
                    f"\n[Round {round_num}] Targets >> Actor: {act_sp:.2%} | Critic: {crit_sp:.2%}"
                )

                current_round_dir = os.path.join(base_exp_dir, f"round_{round_num}")
                os.makedirs(current_round_dir, exist_ok=True)
                mask_path = os.path.join(current_round_dir, "mask.pkl")

                # 1. Create Mask
                if create_ticket:
                    create_mask(
                        cfg,
                        task,
                        seed,
                        ckpt_dir=prev_round_dir,
                        mask_out_path=mask_path,
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
