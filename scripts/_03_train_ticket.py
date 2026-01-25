from __future__ import annotations

import json
import os
import pickle
from typing import Any, Dict, Tuple

from src.agents import SACAgent, SACConfig
from src.data import ReplayBuffer
from src.envs import make_metaworld_env, make_vectorized_metaworld_env
from src.training import run_training_loop, run_vectorized_training_loop
from src.utils import Checkpointer, set_seed
from src.utils.types import Mask


def train_mask(
    cfg: Dict[str, Any],
    task_name: str,
    seed: int,
    mask_path: str,
    save_dir: str,
    rewind_ckpt_path: str,
) -> None:
    """Executes Step 3 of the LTH pipeline: Sparse Ticket Retraining.

    This function performs "Late Rewinding" by resetting the network to a
    pre-convergence checkpoint and applying a pruning mask. This sparse
    configuration is then retrained to evaluate if it can match or exceed
    dense performance.

    Args:
        cfg: Global configuration dictionary.
        task_name: Name of the Meta-World task (e.g., 'reach-v2').
        seed: Random seed for environment and weight initialization.
        mask_path: Path to the .pkl file containing actor and critic masks.
        save_dir: Directory to save checkpoints and metrics for this ticket run.
        rewind_ckpt_path: Explicit path to the rewind checkpoint (the "anchor").

    Raises:
        FileNotFoundError: If the mask file or rewind checkpoint is missing.
        ValueError: If the rewind checkpoint fails to restore.
    """

    # Check for previous completion
    if os.path.exists(os.path.join(save_dir, "training_stats.json")):
        print(f"  [Skip] Ticket training already completed for {task_name} seed {seed}")
        return

    if not os.path.exists(mask_path):
        raise FileNotFoundError(
            f"Mask file not found: {mask_path}. Run create_mask.py first."
        )

    print(f"  > Starting Ticket Training (Sparse)...")

    # 1. Setup Configuration
    hp: Dict[str, Any] = cfg["hyperparameters"]
    defaults: Dict[str, Any] = hp["defaults"]
    task_overrides: Dict[str, Any] = hp.get("tasks", {}).get(task_name, {})
    params: Dict[str, Any] = {**defaults, **task_overrides}

    hidden_dims: Tuple[int, ...] = tuple(hp["hidden_dims"])

    # Parallelization settings
    parallel_config: Dict[str, Any] = cfg["environments"].get("parallel", {})
    use_parallel: bool = parallel_config.get("enabled", False)
    num_envs: int = parallel_config.get("num_envs", 8)
    strategy: str = parallel_config.get("strategy", "sync")

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
        num_envs = 1

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

    # 4. Load the Masks
    print(f"    Loading mask from: {mask_path}")
    with open(mask_path, "rb") as f:
        mask_data: Dict[str, Any] = pickle.load(f)
        actor_mask: Mask = mask_data["actor"]
        critic_mask: Mask = mask_data["critic"]
        sparsity: float = mask_data.get("sparsity_target", 0.0)

    # 5. REWINDING
    rewind_dir: str = os.path.dirname(rewind_ckpt_path)
    rewind_filename: str = os.path.basename(rewind_ckpt_path)
    loader = Checkpointer(rewind_dir)

    dense_state = loader.restore(agent.state, item=rewind_filename)

    if dense_state is None:
        raise ValueError(f"Failed to restore rewind checkpoint from {rewind_ckpt_path}")

    print(f"    Rewinding weights to anchor...")

    # Copy parameters from the anchor state to the current agent
    agent.state = agent.state.replace(
        actor_params=dense_state.actor_params,
        critic_params=dense_state.critic_params,
        target_critic_params=dense_state.target_critic_params,
        log_alpha=dense_state.log_alpha,
    )

    # 6. APPLY MASK
    print(f"    Applying mask (Sparsity ~{sparsity:.0%})...")
    agent.apply_mask(actor_mask, critic_mask)

    # 7. Infrastructure
    buffer = ReplayBuffer(
        obs_dim, act_dim, max_size=hp.get("replay_buffer_size", 1_000_000)
    )
    ticket_checkpointer = Checkpointer(save_dir)

    # 8. Run Training (Sparse)
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
            scale_factor=params.get("scale_rewards", 1.0),
            target_mean_success=params.get("target_mean_success"),
            patience=params.get("patience", 20),
            updates_per_step=params.get("updates_per_step", 1),
            eval_episodes=hp.get("eval_episodes", 5),
            checkpointer=ticket_checkpointer,
            rewind_steps=0,
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
            target_mean_success=params.get("target_mean_success"),
            patience=params.get("patience", 20),
            updates_per_step=1,
            checkpointer=ticket_checkpointer,
            rewind_steps=0,
        )

    # 9. Cleanup and Logging
    print(f"  > Saving replay buffer for subsequent pruning rounds...")
    buffer_data = buffer.save()
    with open(os.path.join(save_dir, "replay_buffer.pkl"), "wb") as f:
        pickle.dump(buffer_data, f)

    env.close()

    with open(os.path.join(save_dir, "training_stats.json"), "w") as f:
        serializable_stats = {k: v for k, v in stats.items() if k != "metrics_history"}
        json.dump(serializable_stats, f, indent=2)

    print(f"  > Ticket training complete. Best success: {stats['best_success']:.2%}")
