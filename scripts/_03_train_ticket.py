"""
Step 3: Train the Lottery Ticket (Sparse Network).

This module defines the 'train_mask' function called by run_pipeline.py.
It implements the "Late Rewinding" logic:
1. Load Binary Mask.
2. Load the specific Rewind Weights (e.g., from step 20k) explicitly.
3. Apply mask immediately to zero out weights.
4. Train with 'use_masking=True' to keep weights sparse.
"""

import json
import os
import pickle
from typing import Any, Dict

import yaml

from src.agents import SACAgent, SACConfig
from src.data import ReplayBuffer
from src.envs import make_metaworld_env, make_vectorized_metaworld_env
from src.training import run_training_loop, run_vectorized_training_loop
from src.utils import Checkpointer, set_seed

# Removed unused matplotlib import
# from matplotlib.pyplot import sca


def train_mask(
    cfg: Dict[str, Any],
    task_name: str,
    seed: int,
    mask_path: str,
    save_dir: str,
    rewind_ckpt_path: str,  # <--- CRITICAL UPDATE: Explicit path to rewind weights
) -> None:
    """
    Executes Step 3 of the LTH pipeline (The Retraining/Ticket Run).

    Args:
        cfg: Config dict.
        task_name: e.g. 'reach-v2'.
        seed: Random seed.
        mask_path: Path to the .pkl file containing actor/critic masks.
        save_dir: Where to save the results of this training run.
        rewind_ckpt_path: Path to the checkpoint to rewind to (e.g., 'round_0/checkpoint_rewind.pkl').
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
            scale_factor=params["scale_rewards"],
            num_envs=num_envs,
            strategy=strategy,
            base_seed=seed,
        )
        print(f"    ✓ Parallelization enabled: {num_envs}x speedup expected")
    else:
        env, obs_dim, act_dim, act_low, act_high = make_metaworld_env(
            task_name, hp["max_episode_steps"], params["scale_rewards"], seed
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

    # 4. Load the Masks (CRITICAL: Must be done before rewinding logic)
    print(f"    Loading mask from: {mask_path}")
    with open(mask_path, "rb") as f:
        mask_data = pickle.load(f)
        actor_mask = mask_data["actor"]
        critic_mask = mask_data["critic"]
        # Use defaults if sparsity metadata isn't present
        sparsity = mask_data.get("sparsity_target", 0.0)

    # 5. REWINDING: Load the specific checkpoint (Winning Ticket Initialization)
    if not os.path.exists(rewind_ckpt_path):
        raise FileNotFoundError(f"Rewind checkpoint not found at: {rewind_ckpt_path}")

    print(f"    Rewinding weights to anchor: {rewind_ckpt_path}")

    # We use the Checkpointer to load the state, but we point it to the directory
    # containing the rewind file.
    rewind_dir = os.path.dirname(rewind_ckpt_path)
    rewind_filename = os.path.basename(rewind_ckpt_path)

    # Initialize a temporary checkpointer just for loading
    loader = Checkpointer(rewind_dir)
    restored_state = loader.restore(agent.state, item=rewind_filename)

    if restored_state is None:
        raise ValueError(f"Failed to restore rewind checkpoint from {rewind_ckpt_path}")

    agent.state = restored_state

    # 6. APPLY MASK
    # Now that weights are reset to T=k, we must zero out the pruned weights.
    # Because use_masking=True, this also registers them as non-trainable.
    print(f"    Applying mask (Sparsity ~{sparsity:.0%})...")
    agent.apply_mask(actor_mask, critic_mask)

    # 7. Infrastructure
    buffer = ReplayBuffer(
        obs_dim, act_dim, max_size=hp.get("replay_buffer_size", 1_000_000)
    )

    ticket_checkpointer = Checkpointer(save_dir)

    # 8. Run Training (Sparse)
    # Note: We pass rewind_steps=0 because we don't want to save a NEW rewind anchor
    # during this retraining phase.

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
            task_name=task_name,
            num_envs=num_envs,
            scale_factor=params.get("scale_rewards", 1),
            target_mean_success=params.get("target_mean_success", None),
            patience=params.get("patience", 20),
            updates_per_step=params.get("updates_per_step", 1),
            eval_episodes=hp.get("eval_episodes", 5),
            checkpointer=ticket_checkpointer,
            rewind_steps=0,  # <--- Do not save rewind anchor in ticket runs
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
            task_name=task_name,
            target_mean_success=params.get("target_mean_success", None),
            patience=params.get("patience", 20),
            updates_per_step=1,
            checkpointer=ticket_checkpointer,
            rewind_steps=0,  # <--- Do not save rewind anchor in ticket runs
        )

    print(f"  > Saving replay buffer (required for next pruning round)...")
    buffer_data = buffer.save()

    # We use 'replay_buffer.pkl' standard name so create_mask can find it
    with open(os.path.join(save_dir, "replay_buffer.pkl"), "wb") as f:
        pickle.dump(buffer_data, f)
    # 9. Cleanup
    env.close()

    # Save Results
    with open(os.path.join(save_dir, "training_stats.json"), "w") as f:
        serializable_stats = {k: v for k, v in stats.items() if k != "metrics_history"}
        json.dump(serializable_stats, f, indent=2)

    print(f"  > Ticket training complete. Best success: {stats['best_success']:.2%}")
