from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from tracemalloc import start
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

from src.training.evaluation import evaluate

if TYPE_CHECKING:
    from src.agents.sac import SACAgent
    from src.data.replay_buffer import ReplayBuffer
    from src.utils.checkpointing import Checkpointer


class Color:
    """ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    END = "\033[0m"


def append_to_jsonl(path: str, data: Dict[str, Any]):
    """Helper to append a dictionary as a JSON line to a file."""
    # Ensure values are serializable (convert numpy types to python native)
    clean_data = {
        k: (
            float(v)
            if isinstance(v, (np.float32, np.float64))
            else int(v) if isinstance(v, (np.int32, np.int64)) else v
        )
        for k, v in data.items()
    }
    with open(path, "a") as f:
        f.write(json.dumps(clean_data) + "\n")


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
    """Core training loop for SAC agent with LTH support."""
    start_time = time.time()

    # 0. Setup Logging Files
    # We save these incrementally so data isn't lost if the run crashes
    os.makedirs(save_dir, exist_ok=True)
    metrics_path = os.path.join(save_dir, "metrics.jsonl")
    episodes_path = os.path.join(save_dir, "train_episodes.jsonl")

    # Wipe files if they exist (start fresh for this run ID)
    # If you want to resume runs, you'd check for existence instead.
    if os.path.exists(metrics_path):
        os.remove(metrics_path)
    if os.path.exists(episodes_path):
        os.remove(episodes_path)

    # Reset environment
    obs, _ = env.reset(seed=seed)

    # Episode tracking
    episode_return = 0.0
    episode_success = 0.0
    episode_len = 0

    # Best performance tracking
    best_return = float("-inf")
    best_success = 0.0

    # Rolling window for console print
    recent_successes = []
    recent_returns = []

    # Early stopping tracking
    no_improve_count = 0
    success_streak = 0
    stop_reason = "max_steps"

    # Metric tracking (for final return)
    metrics_history = []
    # Accumulate training losses between evaluations
    train_metrics_buffer = defaultdict(list)

    step = 0

    for step in range(1, total_steps + 1):
        # 1. Action Selection
        if step < start_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, eval_mode=False)

        # 2. Environment Step
        next_obs, reward, done, truncated, info = env.step(action)
        terminal = done or truncated

        # 3. Store Transition
        replay_buffer.store(obs, action, reward, next_obs, float(done))

        # Update tracking
        obs = next_obs
        episode_return += reward
        episode_success = max(episode_success, float(info.get("success", 0.0)))
        episode_len += 1

        # 4. Handle Episode End
        if terminal:
            # A. Console Logging
            recent_successes.append(episode_success)
            recent_returns.append(episode_return)
            if len(recent_successes) > 100:
                recent_successes.pop(0)

            color = Color.GREEN if episode_success >= 1.0 else Color.RED
            if episode_success >= 1.0:
                print(
                    f"Step {step} | Return: {episode_return:.1f} | "
                    f"{color}Success: {episode_success}{Color.END} | "
                    f"Avg Succ (100): {np.mean(recent_successes[-100:]):.2f} | "
                    f"Avg Ret (100): {np.mean(recent_returns[-100:]):.1f} | "
                    f"Elapsed: {(time.time() - start_time)/60:.1f}min"
                )

            # B. Disk Logging (High Frequency) [CRITICAL ADDITION]
            # This captures exactly when the agent solved the task during training
            episode_log = {
                "step": step,
                "episode_return": episode_return,
                "episode_success": episode_success,
                "episode_length": episode_len,
                "wall_time": time.time() - start_time,
            }
            append_to_jsonl(episodes_path, episode_log)

            # Reset
            obs, _ = env.reset()
            episode_return = 0.0
            episode_success = 0.0
            episode_len = 0

        # 5. Training Updates
        if step >= start_steps and replay_buffer.is_ready(batch_size):
            for _ in range(updates_per_step):
                metrics = agent.update(replay_buffer, batch_size)
                # Buffer metrics for logging
                for k, v in metrics.items():
                    train_metrics_buffer[k].append(float(v))

        # 6. Evaluation & Logging
        if step % eval_interval == 0:
            # Evaluate
            eval_metrics = evaluate(env, agent, num_episodes=eval_episodes)

            # Aggregate training metrics
            avg_train_metrics = {
                f"train/{k}": np.mean(v) if v else 0.0
                for k, v in train_metrics_buffer.items()
            }
            train_metrics_buffer = defaultdict(list)

            # Combine all metrics
            current_log = {
                "step": step,
                "wall_time": time.time() - start_time,
                **eval_metrics,
                **avg_train_metrics,
            }
            metrics_history.append(current_log)

            # C. Disk Logging (Low Frequency) [CRITICAL ADDITION]
            append_to_jsonl(metrics_path, current_log)

            # Console Output
            mean_ret = eval_metrics["mean_return"]
            mean_succ = eval_metrics["mean_success"]

            print(
                f"[Eval] Step {step:>7} | "
                f"Ret: {mean_ret:>6.1f} | "
                f"Succ: {mean_succ:>4.2f} | "
                f"CriticL: {avg_train_metrics.get('train/critic_loss', 0):.2f} | "
                f"Alpha: {agent.alpha:.3f}"
            )

            # Save Checkpoints
            if mean_ret > best_return:
                best_return = mean_ret
                best_success = mean_succ
                no_improve_count = 0

                checkpointer.save(
                    agent.state,
                    filename="checkpoint_best",
                )
                print(f"  {Color.BLUE}>> New Best Model Saved{Color.END}")
            else:
                no_improve_count += 1

            # Early Stopping
            if target_mean_success is not None:
                if mean_succ >= target_mean_success:
                    success_streak += 1
                else:
                    success_streak = 0

                if success_streak >= patience:
                    print(f"{Color.GREEN}>> Early stopping triggered!{Color.END}")
                    stop_reason = "early_stopping"
                    break

    # 7. Final Save
    print(f"{Color.BLUE}>> Saving Final Model...{Color.END}")
    checkpointer.save(agent.state, filename="checkpoint_final")

    stats = {
        "task": task_name,
        "seed": seed,
        "stop_reason": stop_reason,
        "best_return": best_return,
        "best_success": best_success,
        "wall_time_sec": time.time() - start_time,
        "metrics_history": metrics_history,  # Still return this for convenience
    }

    return stats


def run_training_loop_vectorized(
    env,  # AsyncVectorEnv
    agent,
    replay_buffer,
    total_env_steps: int,
    start_steps: int,
    batch_size: int,
    eval_interval: int,
    save_dir: str,
    seed: int,
    task_name: str,
    max_episode_steps: int = 500,
    target_mean_success: Optional[float] = None,
    patience: int = 10,
    updates_per_step: int = 1,
    eval_episodes: int = 5,
    checkpointer=None,
):
    start_time = time.time()

    os.makedirs(save_dir, exist_ok=True)
    metrics_path = os.path.join(save_dir, "metrics.jsonl")
    episodes_path = os.path.join(save_dir, "train_episodes.jsonl")

    for p in [metrics_path, episodes_path]:
        if os.path.exists(p):
            os.remove(p)

    # Create a separate single environment for evaluation
    from src.envs import make_metaworld_env
    eval_env, _, _, _, _ = make_metaworld_env(task_name, max_episode_steps, seed + 99999)

    # --- Init ---
    obs, _ = env.reset(seed=seed)
    num_envs = env.num_envs

    episode_return = np.zeros(num_envs)
    episode_success = np.zeros(num_envs)
    episode_len = np.zeros(num_envs)

    best_return = -np.inf
    best_success = 0.0
    success_streak = 0
    stop_reason = "max_steps"

    train_metrics_buffer = defaultdict(list)
    metrics_history = []

    env_steps = 0

    while env_steps < total_env_steps:
        # 1. Action selection
        if env_steps < start_steps:
            actions = np.stack([
                env.single_action_space.sample()
                for _ in range(num_envs)
            ])
        else:
            actions = agent.select_action(obs, eval_mode=False)

        # 2. Step environments
        next_obs, rewards, terminated, truncated, infos = env.step(actions)
        dones = np.logical_or(terminated, truncated)
        
        # Extract success info from each environment
        # In AsyncVectorEnv, infos is a dict where each key maps to arrays/lists
        success_values = []
        if isinstance(infos, dict) and "success" in infos:
            # Success is already an array
            success_values = infos["success"]
        elif isinstance(infos, (tuple, list)):
            # Tuple/list of info dicts (one per env)
            success_values = [info.get("success", 0.0) for info in infos]
        else:
            # Fallback - no success info available
            success_values = [0.0] * num_envs

        # 3. Store transitions
        for i in range(num_envs):
            replay_buffer.store(
                obs[i],
                actions[i],
                rewards[i],
                next_obs[i],
                float(dones[i]),
            )

        # 4. Episode bookkeeping
        for i in range(num_envs):
            episode_return[i] += rewards[i]
            episode_success[i] = max(
                episode_success[i],
                float(success_values[i]),
            )
            episode_len[i] += 1

            if dones[i]:
                append_to_jsonl(episodes_path, {
                    "step": env_steps,
                    "episode_return": episode_return[i],
                    "episode_success": episode_success[i],
                    "episode_length": episode_len[i],
                    "env_id": i,
                    "wall_time": time.time() - start_time,
                })

                episode_return[i] = 0.0
                episode_success[i] = 0.0
                episode_len[i] = 0

        obs = next_obs
        env_steps += num_envs

        # 5. Training updates
        if env_steps >= start_steps and replay_buffer.is_ready(batch_size):
            for _ in range(updates_per_step * num_envs):
                metrics = agent.update(replay_buffer, batch_size)
                for k, v in metrics.items():
                    train_metrics_buffer[k].append(float(v))

        # 6. Evaluation
        if env_steps % eval_interval < num_envs:
            eval_metrics = evaluate(
                eval_env,
                agent=agent,
                num_episodes=eval_episodes,
            )

            avg_train_metrics = {
                f"train/{k}": np.mean(v) if v else 0.0
                for k, v in train_metrics_buffer.items()
            }
            train_metrics_buffer.clear()

            log = {
                "step": env_steps,
                "wall_time": time.time() - start_time,
                **eval_metrics,
                **avg_train_metrics,
            }
            append_to_jsonl(metrics_path, log)
            metrics_history.append(log)

            mean_ret = eval_metrics["mean_return"]
            mean_succ = eval_metrics["mean_success"]

            print(
                f"[Eval] Step {env_steps:>7} | "
                f"Ret {mean_ret:>6.1f} | "
                f"Succ {mean_succ:>4.2f} | "
                f"Alpha {agent.alpha:.3f}"
            )

            if mean_ret > best_return:
                best_return = mean_ret
                best_success = mean_succ
                success_streak = 0
                checkpointer.save(agent.state, "checkpoint_best")
            else:
                success_streak += 1

            if target_mean_success and success_streak >= patience:
                stop_reason = "early_stopping"
                break

    checkpointer.save(agent.state, "checkpoint_final")
    
    # Clean up eval environment
    eval_env.close()

    return {
        "task": task_name,
        "seed": seed,
        "stop_reason": stop_reason,
        "best_return": best_return,
        "best_success": best_success,
        "wall_time_sec": time.time() - start_time,
        "metrics_history": metrics_history,
    }