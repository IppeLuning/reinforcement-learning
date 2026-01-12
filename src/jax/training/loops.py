"""Training loops for single-task and multi-task SAC.

DISCLAIMER: This code was written by Claude Opus 4.5 on 2026-01-12
and reviewed by Marinus van den Ende.

This module provides the main training orchestration, including:
- Environment interaction
- Replay buffer management
- Periodic evaluation
- Checkpointing
- Early stopping
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

import numpy as np

from src.jax.agents.sac import SACAgent, SACConfig
from src.jax.buffers.replay_buffer import ReplayBuffer
from src.jax.training.evaluation import evaluate
from src.jax.utils.checkpointing import Checkpointer

# Import environment utilities from existing codebase
from src.envs.envs import make_metaworld_env
from src.envs.multitask_wrapper import MultiTaskEnv
from src.utils.seed import set_seed


class Color:
    """ANSI color codes for terminal output."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    END = "\033[0m"


def run_training_loop(
    env: Any,
    agent: SACAgent,
    replay_buffer: ReplayBuffer,
    total_steps: int,
    start_steps: int,
    batch_size: int,
    eval_interval: int,
    save_dir: str,
    seed: int,
    task_name: str,
    target_mean_success: Optional[float] = None,
    patience: int = 10,
    updates_per_step: int = 1,
    eval_episodes: int = 5,
    checkpointer: Optional[Checkpointer] = None,
) -> Dict[str, Any]:
    """Core training loop for SAC agent.
    
    Implements the standard RL training loop with:
    - Random exploration during initial steps
    - Periodic evaluation and checkpointing
    - Early stopping based on success rate
    
    Args:
        env: Gymnasium environment.
        agent: SAC agent to train.
        replay_buffer: Replay buffer for experience storage.
        total_steps: Maximum training steps.
        start_steps: Steps of random exploration before learning.
        batch_size: Batch size for gradient updates.
        eval_interval: Steps between evaluations.
        save_dir: Directory for saving checkpoints.
        seed: Random seed.
        task_name: Name of the task (for logging).
        target_mean_success: Target success rate for early stopping.
        patience: Number of consecutive successes before stopping.
        updates_per_step: Gradient updates per environment step.
        eval_episodes: Number of episodes per evaluation.
        checkpointer: Optional Orbax checkpointer.
        
    Returns:
        Dictionary containing training statistics.
    """
    start_time = time.time()
    
    # Reset environment
    obs, _ = env.reset(seed=seed)
    
    # Episode tracking
    episode_return = 0.0
    episode_success = 0.0
    episode_len = 0
    
    # Best performance tracking
    best_return = float("-inf")
    best_success = 0.0
    best_step = 0
    
    # Early stopping tracking
    no_improve_count = 0
    success_streak = 0
    stop_reason = "max_steps"
    
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Create checkpointer if not provided
    if checkpointer is None:
        checkpointer = Checkpointer(save_dir)
    
    # Training metrics history
    metrics_history = []
    
    for step in range(1, total_steps + 1):
        # Select action
        if step < start_steps:
            # Random exploration
            action = env.action_space.sample()
        else:
            # Policy action
            action = agent.select_action(obs, eval_mode=False)
        
        # Environment step
        next_obs, reward, done, truncated, info = env.step(action)
        terminal = done or truncated
        
        # Store transition (use done, not terminal, for bootstrapping)
        replay_buffer.store(obs, action, reward, next_obs, float(done))
        
        # Update tracking
        obs = next_obs
        episode_return += reward
        episode_success = max(episode_success, float(info.get("success", 0.0)))
        episode_len += 1
        
        # Handle episode end
        if terminal:
            color = Color.GREEN if episode_success >= 1.0 else Color.RED
            print(
                f"[Seed {seed}] Step {step:>7} | "
                f"Task: {info.get('task_name', task_name)} | "
                f"Return: {episode_return:>7.2f} | "
                f"{color}Success: {episode_success:.2f}{Color.END} | "
                f"Len: {episode_len}"
            )
            
            # Reset episode
            obs, _ = env.reset()
            episode_return = 0.0
            episode_success = 0.0
            episode_len = 0
        
        # Training updates
        if step >= start_steps and replay_buffer.is_ready(batch_size):
            for _ in range(updates_per_step):
                metrics = agent.update(replay_buffer, batch_size)
        
        # Evaluation
        if step % eval_interval == 0:
            eval_metrics = evaluate(env, agent, num_episodes=eval_episodes)
            
            mean_ret = eval_metrics["mean_return"]
            best_ret = eval_metrics["best_return"]
            mean_succ = eval_metrics["mean_success"]
            best_succ = eval_metrics["best_success"]
            
            print(
                f"[Eval] Step {step:>7} | Task: {task_name} | "
                f"Return: {mean_ret:>7.1f} (Best: {best_ret:>7.1f}) | "
                f"Success: {mean_succ:>4.2f} (Best: {best_succ:>4.2f}) | "
                f"Alpha: {agent.alpha:.4f} | "
                f"No Improve: {no_improve_count}"
            )
            
            # Store metrics
            eval_metrics["step"] = step
            eval_metrics["alpha"] = agent.alpha
            metrics_history.append(eval_metrics)
            
            # Check for improvement
            if mean_ret > best_return + 1e-3:
                best_return = mean_ret
                best_success = mean_succ
                best_step = step
                no_improve_count = 0
                
                # Save best model
                checkpointer.save(
                    agent.state,
                    step=step,
                    extra={"task": task_name, "seed": seed, "is_best": True},
                )
                print(f"{Color.BLUE}Saved new best model (Return: {best_return:.2f}){Color.END}")
            else:
                no_improve_count += 1
            
            # Early stopping check
            if target_mean_success is not None:
                if mean_succ >= target_mean_success:
                    success_streak += 1
                else:
                    success_streak = 0
                
                if success_streak >= patience:
                    print(
                        f"{Color.GREEN}[Seed {seed}] Early stopping at step {step} "
                        f"(Success: {mean_succ:.2f}){Color.END}"
                    )
                    stop_reason = "early_stopping"
                    break
    
    # Final statistics
    total_time = time.time() - start_time
    stats = {
        "task": task_name,
        "seed": seed,
        "total_steps_trained": step,
        "stop_reason": stop_reason,
        "best_step": best_step,
        "best_return": best_return,
        "best_success": best_success,
        "wall_time_sec": total_time,
        "steps_per_sec": step / total_time if total_time > 0 else 0,
        "metrics_history": metrics_history,
    }
    
    return stats


def train_single_task_session(
    cfg: Dict[str, Any],
    task_name: str,
    seed: int,
) -> Dict[str, Any]:
    """Train a single-task SAC agent.
    
    This is the main entry point for single-task training sessions.
    Handles all setup, training, and result saving.
    
    Args:
        cfg: Full configuration dictionary.
        task_name: Name of the Meta-World task.
        seed: Random seed for reproducibility.
        
    Returns:
        Dictionary containing training statistics.
    """
    # Set random seeds
    set_seed(seed)
    
    # Extract configuration
    single_cfg = cfg["single_task"]
    defaults = single_cfg.get("defaults", {})
    task_overrides = single_cfg.get("tasks", {}).get(task_name, {})
    
    # Merge defaults with task-specific overrides
    params = {**defaults, **task_overrides}
    params["hidden_dims"] = tuple(cfg["network"]["hidden_dims"])
    
    # Extract training parameters
    target_success = params.pop("target_mean_success", None)
    patience = params.pop("patience", 10)
    updates_per_step = params.pop("updates_per_step", 1)
    
    # Create save path
    save_path = os.path.join(
        cfg["defaults"]["save_dir"],
        "jax",
        task_name,
        f"seed_{seed}",
    )
    os.makedirs(save_path, exist_ok=True)
    
    # Create environment
    env, obs_dim, act_dim, act_low, act_high = make_metaworld_env(
        task_name,
        cfg["defaults"]["max_episode_steps"],
        seed,
    )
    
    # Create SAC config and agent
    sac_config = SACConfig(
        gamma=params.get("gamma", 0.99),
        tau=params.get("tau", 0.005),
        actor_lr=params.get("actor_lr", 3e-4),
        critic_lr=params.get("critic_lr", 3e-4),
        alpha_lr=params.get("alpha_lr", 3e-4),
        target_entropy_scale=params.get("target_entropy_scale", 1.0),
        auto_alpha=params.get("auto_alpha", True),
        init_alpha=params.get("init_alpha", 1.0),
        hidden_dims=params["hidden_dims"],
    )
    
    agent = SACAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_low=act_low,
        act_high=act_high,
        config=sac_config,
        seed=seed,
    )
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(
        obs_dim=obs_dim,
        act_dim=act_dim,
        max_size=single_cfg.get("replay_buffer_size", 1_000_000),
    )
    
    print(f"\n{'='*60}")
    print(f"[JAX Single-Task] {task_name} | Seed {seed}")
    print(f"{'='*60}")
    print(f"  Config: alpha={sac_config.init_alpha}, "
          f"entropy_scale={sac_config.target_entropy_scale}, "
          f"patience={patience}")
    
    # Run training
    stats = run_training_loop(
        env=env,
        agent=agent,
        replay_buffer=replay_buffer,
        total_steps=single_cfg["total_steps"],
        start_steps=single_cfg["start_steps"],
        batch_size=single_cfg["batch_size"],
        eval_interval=single_cfg["eval_interval"],
        save_dir=save_path,
        seed=seed,
        task_name=task_name,
        target_mean_success=target_success,
        patience=patience,
        updates_per_step=updates_per_step,
    )
    
    # Save results
    results_path = os.path.join(save_path, "results.json")
    with open(results_path, "w") as f:
        # Remove non-serializable items
        stats_serializable = {
            k: v for k, v in stats.items() 
            if k != "metrics_history"
        }
        json.dump(
            {"config": cfg, "sac_params": params, "results": stats_serializable},
            f,
            indent=2,
        )
    print(f"Results saved to {results_path}")
    
    env.close()
    return stats


def train_multitask_session(
    cfg: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    """Train a multi-task SAC agent.
    
    Trains a single agent on multiple tasks simultaneously using
    task-augmented observations (one-hot task ID concatenated to obs).
    
    Args:
        cfg: Full configuration dictionary.
        seed: Random seed for reproducibility.
        
    Returns:
        Dictionary containing training statistics.
    """
    # Set random seeds
    set_seed(seed)
    
    # Extract configuration
    multi_cfg = cfg["multi_task"]
    params = multi_cfg.get("sac_params", {}).copy()
    params["hidden_dims"] = tuple(cfg["network"]["hidden_dims"])
    
    target_success = params.pop("target_mean_success", None)
    patience = params.pop("patience", 20)
    
    tasks = cfg["environments"]["tasks"]
    task_name_for_log = "MT_shared"
    
    # Create save path
    save_path = os.path.join(
        cfg["defaults"]["save_dir"],
        "jax",
        "multitask",
        f"seed_{seed}",
    )
    os.makedirs(save_path, exist_ok=True)
    
    # Create environments for each task
    envs_dict = {}
    obs_dim, act_dim, act_low, act_high = 0, 0, None, None
    
    for task in tasks:
        env, o, a, low, high = make_metaworld_env(
            task,
            cfg["defaults"]["max_episode_steps"],
            seed,
        )
        envs_dict[task] = env
        obs_dim, act_dim, act_low, act_high = o, a, low, high
    
    # Wrap in multi-task environment
    mt_env = MultiTaskEnv(
        envs_dict,
        max_episode_steps=cfg["defaults"]["max_episode_steps"],
    )
    
    # Effective observation dimension includes task one-hot
    num_tasks = len(tasks)
    effective_obs_dim = obs_dim + num_tasks
    
    # Create SAC config and agent
    sac_config = SACConfig(
        gamma=params.get("gamma", 0.99),
        tau=params.get("tau", 0.005),
        actor_lr=params.get("actor_lr", 3e-4),
        critic_lr=params.get("critic_lr", 3e-4),
        alpha_lr=params.get("alpha_lr", 3e-4),
        target_entropy_scale=params.get("target_entropy_scale", 1.0),
        auto_alpha=params.get("auto_alpha", True),
        init_alpha=params.get("init_alpha", 1.0),
        hidden_dims=params["hidden_dims"],
    )
    
    agent = SACAgent(
        obs_dim=effective_obs_dim,
        act_dim=act_dim,
        act_low=act_low,
        act_high=act_high,
        config=sac_config,
        seed=seed,
    )
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(
        obs_dim=effective_obs_dim,
        act_dim=act_dim,
        max_size=multi_cfg.get("replay_buffer_size", 2_000_000),
    )
    
    print(f"\n{'='*60}")
    print(f"[JAX Multi-Task] Tasks: {tasks} | Seed {seed}")
    print(f"{'='*60}")
    print(f"  Config: alpha={sac_config.init_alpha}, "
          f"entropy_scale={sac_config.target_entropy_scale}")
    
    # Run training
    stats = run_training_loop(
        env=mt_env,
        agent=agent,
        replay_buffer=replay_buffer,
        total_steps=multi_cfg["total_steps"],
        start_steps=multi_cfg["start_steps"],
        batch_size=multi_cfg["batch_size"],
        eval_interval=multi_cfg["eval_interval"],
        save_dir=save_path,
        seed=seed,
        task_name=task_name_for_log,
        target_mean_success=target_success,
        patience=patience,
        eval_episodes=10,  # More episodes for multi-task
    )
    
    # Save results
    results_path = os.path.join(save_path, "results.json")
    with open(results_path, "w") as f:
        stats_serializable = {
            k: v for k, v in stats.items()
            if k != "metrics_history"
        }
        json.dump(
            {"config": cfg, "sac_params": params, "results": stats_serializable},
            f,
            indent=2,
        )
    print(f"Results saved to {results_path}")
    
    mt_env.close()
    return stats
