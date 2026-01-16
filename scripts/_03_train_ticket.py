"""
Step 3: Train the Lottery Ticket (Sparse Network).

This module defines the 'train_mask' function called by run_pipeline.py.
It implements the "Rewinding" logic:
1. Load Binary Mask.
2. Load W0 (Initial weights) via rewind_to_ticket.
3. Train with 'use_masking=True' to keep weights sparse.
"""

import json
import os
import pickle
from typing import Any, Dict

import yaml

from src.agents import SACAgent, SACConfig
from src.data import ReplayBuffer
from src.envs import make_metaworld_env, make_vectorized_metaworld_env
from src.lth.recovery import rewind_to_ticket
from src.training import run_training_loop, run_vectorized_training_loop
from src.utils import Checkpointer, set_seed


def train_mask(
    cfg: Dict[str, Any], task_name: str, seed: int, mask_path: str, save_dir: str
) -> None:
    """
    Executes Step 3 of the LTH pipeline (The Transfer/Ticket Run).
    """

    # Check completion
    if os.path.exists(os.path.join(save_dir, "training_stats.json")):
        print(f"  [Skip] Ticket training already completed for {task_name} seed {seed}")
        return

    if not os.path.exists(mask_path):
        raise FileNotFoundError(
            f"Mask file not found: {mask_path}. Run create_mask.py first."
        )

    print(f"  > Starting Ticket Training (Sparse)...")

    # 1. Setup Configuration
    hp = cfg["hyperparameters"]
    defaults = hp["defaults"]
    task_overrides = hp.get("tasks", {}).get(task_name, {})
    params = {**defaults, **task_overrides}

    hidden_dims = tuple(hp["hidden_dims"])

    # Check for parallelization settings
    parallel_config = cfg["environments"].get("parallel", {})
    use_parallel = parallel_config.get("enabled", False)
    num_envs = parallel_config.get("num_envs", 8)
    strategy = parallel_config.get("strategy", "sync")

    # 2. Initialize Environment & Seeding
    set_seed(seed)

    if use_parallel:
        print(f"    Creating {num_envs} parallel environments ({strategy} mode)...")
        env, obs_dim, act_dim, act_low, act_high = make_vectorized_metaworld_env(
            task_name=task_name,
            max_episode_steps=hp["max_episode_steps"],
            num_envs=num_envs,
            strategy=strategy,
            base_seed=seed,
        )
        print(f"    ✓ Parallelization enabled: {num_envs}x speedup expected")
    else:
        env, obs_dim, act_dim, act_low, act_high = make_metaworld_env(
            task_name, hp["max_episode_steps"], seed
        )
        num_envs = 1  # For compatibility

    # 3. Initialize Agent with Masking ENABLED
    sac_config = SACConfig(
        gamma=params.get("gamma", 0.99),
        tau=params.get("tau", 0.005),
        actor_lr=params.get("actor_lr", 3e-4),
        critic_lr=params.get("critic_lr", 3e-4),
        alpha_lr=params.get("alpha_lr", 3e-4),
        target_entropy_scale=params.get("target_entropy_scale", 1.0),
        auto_alpha=params.get("auto_alpha", True),
        init_alpha=params.get("init_alpha", 0.2),
        hidden_dims=hidden_dims,
        use_masking=True,
    )

    agent = SACAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_low=act_low,
        act_high=act_high,
        config=sac_config,
        seed=seed,
    )

    # 4. Load the Masks (CRITICAL: Must be done before rewinding)
    print(f"    Loading mask from: {mask_path}")
    with open(mask_path, "rb") as f:
        mask_data = pickle.load(f)
        actor_mask = mask_data["actor"]
        critic_mask = mask_data["critic"]
        # Use defaults if sparsity metadata isn't present
        sparsity = mask_data.get("sparsity_target", 0.0)

    # 5. Rewind to Ticket (W0 * Mask)
    # This assumes the dense training was done on the same seed and saved to standard path.
    # Standard path: data/checkpoints/{task_name}/seed_{seed}
    dense_ckpt_dir = os.path.join("data", "checkpoints", task_name, f"seed_{seed}")

    # NOTE: If doing transfer (using Task A mask on Task B), 'mask_path' comes from A,
    # but 'dense_ckpt_dir' must point to B's initialization to get the correct W0 for this seed.

    rewind_to_ticket(
        agent=agent,
        dense_ckpt_dir=dense_ckpt_dir,
        actor_mask=actor_mask,
        critic_mask=critic_mask,
    )

    print(f"    Ticket successfully constructed (Sparsity ~{sparsity:.0%}).")

    # 6. Infrastructure
    buffer = ReplayBuffer(
        obs_dim, act_dim, max_size=hp.get("replay_buffer_size", 1_000_000)
    )

    ticket_checkpointer = Checkpointer(save_dir)

    # 7. Run Training (Sparse) with or without parallelization
    if use_parallel:
        stats = run_vectorized_training_loop(
            vec_env=env,
            agent=agent,
            replay_buffer=buffer,
            total_steps=hp["total_steps"],
            start_steps=hp["start_steps"],
            batch_size=hp["batch_size"],
            eval_interval=hp["eval_interval"],
            save_dir=save_dir,
            seed=seed,
            task_name=task_name,  # Use original task_name for env creation
            num_envs=num_envs,
            target_mean_success=params.get("target_mean_success", None),
            patience=params.get("patience", 20),
            updates_per_step=params.get("updates_per_step", 1),
            eval_episodes=hp.get("eval_episodes", 5),
            checkpointer=ticket_checkpointer,
            max_episode_steps=hp["max_episode_steps"],
        )
    else:
        stats = run_training_loop(
            env=env,
            agent=agent,
            replay_buffer=buffer,
            total_steps=hp["total_steps"],
            start_steps=hp["start_steps"],
            batch_size=hp["batch_size"],
            eval_interval=hp["eval_interval"],
            save_dir=save_dir,
            seed=seed,
            task_name=task_name,  # Use original task_name for env creation
            target_mean_success=params.get("target_mean_success", None),
            patience=params.get("patience", 20),
            updates_per_step=1,
            checkpointer=ticket_checkpointer,
        )

    # 8. Cleanup
    env.close()

    # Save Results
    with open(os.path.join(save_dir, "training_stats.json"), "w") as f:
        serializable_stats = {k: v for k, v in stats.items() if k != "metrics_history"}
        json.dump(serializable_stats, f, indent=2)

    print(f"  > Ticket training complete. Best success: {stats['best_success']:.2%}")
