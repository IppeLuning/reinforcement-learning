"""
Example script demonstrating parallelized training with vectorized environments.

This script shows how to use the new vectorized environment wrapper to speed up
SAC training by collecting data from multiple environments in parallel.

Expected speedup: 4-8x faster data collection with num_envs=8 (sync mode)
"""

import os
from pathlib import Path

import jax
import yaml

from src.agents.sac import SACAgent
from src.data.replay_buffer import ReplayBuffer
from src.envs.vectorized import make_vectorized_metaworld_env
from src.training.loops_vectorized import run_vectorized_training_loop
from src.utils.checkpointing import Checkpointer
from src.utils.jax_setup import configure_jax


def main():
    # Configure JAX
    configure_jax()

    # Load config
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Extract settings
    task_name = config["environments"]["tasks"][0]
    seed = config["environments"]["seeds"][0]

    # Parallelization settings
    parallel_config = config["environments"].get("parallel", {})
    use_parallel = parallel_config.get("enabled", False)
    num_envs = parallel_config.get("num_envs", 8)
    strategy = parallel_config.get("strategy", "sync")

    # Hyperparameters
    hyper = config["hyperparameters"]
    total_steps = hyper["total_steps"]
    start_steps = hyper["start_steps"]
    batch_size = hyper["batch_size"]
    eval_interval = hyper["eval_interval"]
    max_episode_steps = hyper["max_episode_steps"]
    replay_buffer_size = hyper["replay_buffer_size"]
    hidden_dims = hyper["hidden_dims"]
    updates_per_step = hyper["updates_per_step"]

    # Get task-specific hyperparameters
    defaults = hyper["defaults"]
    task_overrides = hyper["tasks"].get(task_name, {})
    task_hyper = {**defaults, **task_overrides}

    print("=" * 80)
    print(f"PARALLELIZED SAC TRAINING")
    print("=" * 80)
    print(f"Task: {task_name}")
    print(f"Seed: {seed}")
    print(f"Parallelization: {'ENABLED' if use_parallel else 'DISABLED'}")
    if use_parallel:
        print(f"  - Num Envs: {num_envs}")
        print(f"  - Strategy: {strategy}")
        print(f"  - Expected Speedup: ~{num_envs}x for data collection")
    print("=" * 80)

    # Create environment
    if use_parallel:
        print(f"\n🚀 Creating {num_envs} parallel environments ({strategy} mode)...")
        vec_env, obs_dim, act_dim, act_low, act_high = make_vectorized_metaworld_env(
            task_name=task_name,
            max_episode_steps=max_episode_steps,
            num_envs=num_envs,
            strategy=strategy,
            base_seed=seed,
        )
        print(f"✓ Vectorized environment created")
    else:
        print("\n⚠️  Using single environment (slower)")
        from src.envs.factory import make_metaworld_env

        vec_env, obs_dim, act_dim, act_low, act_high = make_metaworld_env(
            task_name, max_episode_steps, seed=seed
        )
        num_envs = 1  # For compatibility

    # Create agent
    print("\n🤖 Creating SAC agent...")
    key = jax.random.PRNGKey(seed)
    agent = SACAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_low=act_low,
        act_high=act_high,
        hidden_dims=hidden_dims,
        gamma=task_hyper["gamma"],
        tau=task_hyper["tau"],
        actor_lr=task_hyper["actor_lr"],
        critic_lr=task_hyper["critic_lr"],
        alpha_lr=task_hyper["alpha_lr"],
        init_alpha=task_hyper["init_alpha"],
        auto_alpha=task_hyper["auto_alpha"],
        target_entropy_scale=task_hyper.get("target_entropy_scale", 1.0),
        key=key,
    )
    print("✓ Agent initialized")

    # Create replay buffer
    print("\n💾 Creating replay buffer...")
    replay_buffer = ReplayBuffer(
        obs_dim=obs_dim,
        act_dim=act_dim,
        max_size=replay_buffer_size,
    )
    print(f"✓ Replay buffer ready (capacity: {replay_buffer_size:,})")

    # Setup checkpointing
    save_dir = f"checkpoints/parallel_demo/{task_name}/seed_{seed}"
    os.makedirs(save_dir, exist_ok=True)
    checkpointer = Checkpointer(save_dir)
    print(f"\n💾 Checkpoints will be saved to: {save_dir}")

    # Run training
    print("\n" + "=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)

    if use_parallel:
        stats = run_vectorized_training_loop(
            vec_env=vec_env,
            agent=agent,
            replay_buffer=replay_buffer,
            total_steps=total_steps,
            start_steps=start_steps,
            batch_size=batch_size,
            eval_interval=eval_interval,
            save_dir=save_dir,
            seed=seed,
            task_name=task_name,
            num_envs=num_envs,
            target_mean_success=task_hyper.get("target_mean_success"),
            patience=task_hyper.get("patience", 10),
            updates_per_step=updates_per_step,
            eval_episodes=5,
            checkpointer=checkpointer,
        )
    else:
        from src.training.loops import run_training_loop

        stats = run_training_loop(
            env=vec_env,
            agent=agent,
            replay_buffer=replay_buffer,
            total_steps=total_steps,
            start_steps=start_steps,
            batch_size=batch_size,
            eval_interval=eval_interval,
            save_dir=save_dir,
            seed=seed,
            task_name=task_name,
            target_mean_success=task_hyper.get("target_mean_success"),
            patience=task_hyper.get("patience", 10),
            updates_per_step=updates_per_step,
            eval_episodes=5,
            checkpointer=checkpointer,
        )

    # Print final stats
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Stop Reason: {stats['stop_reason']}")
    print(f"Best Return: {stats['best_return']:.2f}")
    print(f"Best Success: {stats['best_success']:.2%}")
    print(f"Wall Time: {stats['wall_time_sec'] / 60:.1f} minutes")
    if use_parallel:
        print(f"Num Environments: {num_envs}")
        print(f"Effective Sample Rate: ~{num_envs}x faster than single env")
    print("=" * 80)


if __name__ == "__main__":
    main()
