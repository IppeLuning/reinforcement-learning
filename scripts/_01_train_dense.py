from __future__ import annotations

import json
import os
import pickle
from typing import Any, Dict, Optional, Tuple

from src.agents import SACAgent, SACConfig
from src.data import ReplayBuffer
from src.envs import make_metaworld_env, make_vectorized_metaworld_env
from src.training import run_training_loop, run_vectorized_training_loop
from src.utils import Checkpointer, set_seed


def train_dense(
    cfg: Dict[str, Any], task_name: str, seed: int, save_dir: str, rewind_steps: int
) -> None:
    """Executes Step 1 of the LTH pipeline: Dense Training and W0 capture.

    This function performs the following sequence:
    1. Initializes the environment and network with a specific seed.
    2. Captures the initial weight state ($W_0$) as the 'winning ticket' baseline.
    3. Trains the model to convergence using SAC.
    4. Saves the final trained weights and the replay buffer for pruning analysis.

    Args:
        cfg: The global configuration dictionary containing hyperparameters and environment settings.
        task_name: Name of the Meta-World task to train on.
        seed: Random seed for environment and weight initialization.
        save_dir: Directory where checkpoints and logs will be stored.
        rewind_steps: Global step at which to capture a specific rewind checkpoint ($W_{rewind}$).

    Returns:
        None. All outputs are saved to the `save_dir`.
    """

    if os.path.exists(os.path.join(save_dir, "checkpoint_final.pkl")):
        print(f"  [Skip] Dense training already completed for {task_name} seed {seed}")
        return

    print(f"  > Starting Dense Training...")

    # 1. Setup Configuration
    hp: Dict[str, Any] = cfg["hyperparameters"]
    defaults: Dict[str, Any] = hp["defaults"]
    task_overrides: Dict[str, Any] = hp.get("tasks", {}).get(task_name, {})

    # Merge: Overrides > Defaults
    params: Dict[str, Any] = {**defaults, **task_overrides}

    hidden_dims: Tuple[int, ...] = tuple(hp["hidden_dims"])
    total_steps: int = hp["total_steps"]
    start_steps: int = hp["start_steps"]

    # Parallelization settings
    parallel_config: Dict[str, Any] = cfg["environments"].get("parallel", {})
    use_parallel: bool = parallel_config.get("enabled", False)
    num_envs: int = parallel_config.get("num_envs", 8)
    strategy: str = parallel_config.get("strategy", "sync")

    # 2. Initialize Environment & Seeding
    set_seed(seed)

    if use_parallel:
        print(f"  > Creating {num_envs} parallel environments ({strategy} mode)...")
        env, obs_dim, act_dim, act_low, act_high = make_vectorized_metaworld_env(
            task_name=task_name,
            max_episode_steps=hp["max_episode_steps"],
            scale_factor=params.get("scale_rewards", 10.0),
            num_envs=num_envs,
            strategy=strategy,
            base_seed=seed,
        )
        print(f"  ✓ Parallelization enabled: {num_envs}x speedup expected")
    else:
        env, obs_dim, act_dim, act_low, act_high = make_metaworld_env(
            task_name, hp["max_episode_steps"], params.get("scale_rewards", 10.0), seed
        )
        num_envs = 1

    # 3. Initialize Agent
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
        use_masking=False,
    )

    agent = SACAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_low=act_low,
        act_high=act_high,
        config=sac_config,
        seed=seed,
    )

    # 4. Infrastructure
    buffer = ReplayBuffer(
        obs_dim, act_dim, max_size=hp.get("replay_buffer_size", 1_000_000)
    )
    checkpointer = Checkpointer(save_dir)

    # 5. [CRITICAL LTH STEP] Save Initial Weights (W0)
    print(f"  > Saving W0 (checkpoint_init.pkl)...")
    checkpointer.save(agent.state, filename="checkpoint_init")

    # 6. Run Training
    if use_parallel:
        stats = run_vectorized_training_loop(
            vec_env=env,
            agent=agent,
            replay_buffer=buffer,
            total_steps=total_steps,
            start_steps=start_steps,
            batch_size=hp["batch_size"],
            eval_interval=hp["eval_interval"],
            save_dir=save_dir,
            seed=seed,
            task_name=task_name,
            num_envs=num_envs,
            scale_factor=params.get("scale_rewards", 10.0),
            target_mean_success=params.get("target_mean_success"),
            patience=params.get("patience", 20),
            updates_per_step=params.get("updates_per_step", 1),
            eval_episodes=hp.get("eval_episodes", 5),
            checkpointer=checkpointer,
            max_episode_steps=hp["max_episode_steps"],
            rewind_steps=rewind_steps,
        )
    else:
        stats = run_training_loop(
            env=env,
            agent=agent,
            replay_buffer=buffer,
            total_steps=total_steps,
            start_steps=start_steps,
            batch_size=hp["batch_size"],
            eval_interval=hp["eval_interval"],
            save_dir=save_dir,
            seed=seed,
            task_name=task_name,
            target_mean_success=params.get("target_mean_success"),
            patience=params.get("patience", 20),
            updates_per_step=params.get("updates_per_step", 1),
            eval_episodes=hp.get("eval_episodes", 5),
            checkpointer=checkpointer,
            rewind_steps=rewind_steps,
        )

    # 7. Save replay buffer for gradient-based saliency calculations
    print(f"  > Saving replay buffer...")
    buffer_data: Dict[str, Any] = buffer.save()
    with open(os.path.join(save_dir, "replay_buffer.pkl"), "wb") as f:
        pickle.dump(buffer_data, f)

    # 8. Cleanup and Logging
    env.close()

    with open(os.path.join(save_dir, "training_stats.json"), "w") as f:
        serializable_stats = {k: v for k, v in stats.items() if k != "metrics_history"}
        json.dump(serializable_stats, f, indent=2)

    print(f"  > Dense training complete. Best success: {stats['best_success']:.2%}")
