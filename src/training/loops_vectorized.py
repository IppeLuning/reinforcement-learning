"""Training loops supporting both single and vectorized environments."""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from encodings import hp_roman8
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


def run_vectorized_training_loop(
    vec_env: Any,  # VectorizedMetaWorldEnv
    agent: SACAgent,
    replay_buffer: ReplayBuffer,
    total_steps: int,
    start_steps: int,
    batch_size: int,
    eval_interval: int,
    save_dir: str,
    seed: int,
    task_name: str,
    num_envs: int,
    scale_factor: float,
    target_mean_success: Optional[float] = None,
    patience: int = 10,
    updates_per_step: int = 1,
    eval_episodes: int = 5,
    checkpointer: Optional[Checkpointer] = None,
    rewind_steps: int = 0,
    max_episode_steps: int = 500,
    resume_from_step: int = 0,
) -> Dict[str, Any]:
    """
    Training loop optimized for vectorized environments.

    Collects data from multiple environments in parallel, dramatically
    speeding up the data collection phase of SAC training.

    Key differences from single-env loop:
    - Steps `num_envs` environments simultaneously
    - Tracks multiple episodes at once
    - Effective sample collection rate is `num_envs * steps_per_second`
    
    Args:
        resume_from_step: If > 0, resume training from this step (checkpoint already loaded).
    """
    start_time = time.time()

    # Setup Logging Files
    os.makedirs(save_dir, exist_ok=True)
    metrics_path = os.path.join(save_dir, "metrics.jsonl")
    episodes_path = os.path.join(save_dir, "train_episodes.jsonl")

    # Only wipe files if starting fresh (not resuming)
    if resume_from_step == 0:
        if os.path.exists(metrics_path):
            os.remove(metrics_path)
        if os.path.exists(episodes_path):
            os.remove(episodes_path)
    else:
        print(f"{Color.BLUE}>> Resuming training from step {resume_from_step}{Color.END}")

    # Reset all environments
    obs, _ = vec_env.reset(seed=seed)

    # Per-environment episode tracking
    episode_returns = np.zeros(num_envs)
    episode_successes = np.zeros(num_envs)
    episode_lens = np.zeros(num_envs, dtype=int)

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

    # Metric tracking
    metrics_history = []
    train_metrics_buffer = defaultdict(list)

    # Total environment steps (across all parallel envs)
    # If resuming, start from the checkpoint step
    total_env_steps = resume_from_step + num_envs if resume_from_step > 0 else num_envs
    step = 0
    rewind_saved = resume_from_step >= rewind_steps if rewind_steps > 0 else False

    print(f"{Color.BLUE}>> Training with {num_envs} parallel environments{Color.END}")

    while total_env_steps < total_steps:
        step += 1

        # 1. Action Selection (for all environments)
        if total_env_steps < start_steps:
            # Random actions for each environment
            actions = np.array([vec_env.action_space.sample() for _ in range(num_envs)])
        else:
            # Agent selects actions for each environment
            actions = np.array(
                [agent.select_action(obs[i], eval_mode=False) for i in range(num_envs)]
            )

        # 2. Step all environments in parallel
        next_obs, rewards, dones, truncateds, infos = vec_env.step(actions)
        terminals = np.logical_or(dones, truncateds)

        # 3. Store transitions for all environments
        for i in range(num_envs):
            replay_buffer.store(
                obs[i], actions[i], rewards[i], next_obs[i], float(dones[i])
            )

            # Update episode tracking
            episode_returns[i] += rewards[i]
            episode_lens[i] += 1

            # Track success if available in current step's info
            # In vectorized envs, info structure can vary:
            # - dict with "final_info" key containing list of final infos
            # - dict with arrays for each info field
            # - tuple/list of dicts
            try:
                if isinstance(infos, dict):
                    # Check for final_info structure (newer Gymnasium)
                    if "final_info" in infos and i < len(infos["final_info"]):
                        final_info = infos["final_info"][i]
                        if final_info is not None and "success" in final_info:
                            episode_successes[i] = max(
                                episode_successes[i], float(final_info["success"])
                            )
                    # Check for direct success array
                    elif "success" in infos and hasattr(
                        infos["success"], "__getitem__"
                    ):
                        episode_successes[i] = max(
                            episode_successes[i], float(infos["success"][i])
                        )
                elif isinstance(infos, (list, tuple)) and i < len(infos):
                    # Old style: list/tuple of dicts
                    if isinstance(infos[i], dict) and "success" in infos[i]:
                        episode_successes[i] = max(
                            episode_successes[i], float(infos[i]["success"])
                        )
            except (KeyError, IndexError, TypeError):
                # If we can't extract success, just continue
                # Success will be 0.0 which is the default
                pass

        # Update counters
        total_env_steps += num_envs
        obs = next_obs

        if rewind_steps > 0 and not rewind_saved and total_env_steps >= rewind_steps:
            print(
                f"{Color.BLUE}>> Saving Rewind Weights (Step {total_env_steps})...{Color.END}"
            )
            # Don't save optimizer state for rewind - we want fresh optimizers when rewinding
            checkpointer.save(agent.state, filename="checkpoint_rewind", save_optimizer=False)
            rewind_saved = True

        # 4. Handle episode endings
        for i in range(num_envs):
            if terminals[i]:
                # Console and disk logging
                recent_successes.append(episode_successes[i])
                recent_returns.append(episode_returns[i])
                if len(recent_successes) > 100:
                    recent_successes.pop(0)
                    recent_returns.pop(0)

                # Print for ALL episode completions (matching original loop behavior)
                color = Color.GREEN if episode_successes[i] >= 1.0 else Color.RED
                print(
                    f"Step {total_env_steps} | Env {i} | Return: {episode_returns[i]:.1f} | "
                    f"{color}Success: {episode_successes[i]:.2f}{Color.END} | "
                    f"Avg Succ (100): {np.mean(recent_successes[-100:]):.2f} | "
                    f"Avg Ret (100): {np.mean(recent_returns[-100:]):.1f} | "
                    f"Elapsed: {(time.time() - start_time)/60:.1f}min"
                )

                # Disk logging
                episode_log = {
                    "step": total_env_steps,
                    "env_id": i,
                    "episode_return": float(episode_returns[i]),
                    "episode_success": float(episode_successes[i]),
                    "episode_length": int(episode_lens[i]),
                    "wall_time": time.time() - start_time,
                }
                append_to_jsonl(episodes_path, episode_log)

                # Reset tracking for this environment
                episode_returns[i] = 0.0
                episode_successes[i] = 0.0
                episode_lens[i] = 0

        # 5. Training Updates
        # With vectorized envs, we collect data faster, so we might want more updates per step
        if total_env_steps >= start_steps and replay_buffer.is_ready(batch_size):
            # Scale updates by number of environments to maintain update-to-data ratio
            num_updates = int(updates_per_step * num_envs)

            # Optimization: Do all updates without intermediate Python calls
            for _ in range(num_updates):
                metrics = agent.update(replay_buffer, batch_size)
                for k, v in metrics.items():
                    train_metrics_buffer[k].append(float(v))

        # 6. Evaluation & Logging
        if (
            total_env_steps >= eval_interval
            and total_env_steps % eval_interval < num_envs
        ):
            # Create a single eval environment (not vectorized)
            from src.envs.factory import make_metaworld_env

            eval_env, _, _, _, _ = make_metaworld_env(
                task_name, max_episode_steps, scale_factor, seed=seed
            )

            # Evaluate
            eval_metrics = evaluate(eval_env, agent, num_episodes=eval_episodes)
            eval_env.close()

            # Aggregate training metrics
            avg_train_metrics = {
                f"train/{k}": np.mean(v) if v else 0.0
                for k, v in train_metrics_buffer.items()
            }
            train_metrics_buffer = defaultdict(list)

            # Combine all metrics
            current_log = {
                "step": total_env_steps,
                "wall_time": time.time() - start_time,
                **eval_metrics,
                **avg_train_metrics,
            }
            metrics_history.append(current_log)

            # Disk logging
            append_to_jsonl(metrics_path, current_log)

            # Console output
            mean_ret = eval_metrics["mean_return"]
            mean_succ = eval_metrics["mean_success"]

            print(
                f"[Eval] Step {total_env_steps:>7} | "
                f"Ret: {mean_ret:>6.1f} | "
                f"Succ: {mean_succ:>4.2f} | "
                f"CriticL: {avg_train_metrics.get('train/critic_loss', 0):.2f} | "
                f"Alpha: {agent.alpha:.3f}"
            )

            checkpointer.save(
                agent.state,
                filename=f"checkpoint_step_{total_env_steps}",
            )

            if mean_ret > best_return:
                best_return = mean_ret
                best_success = mean_succ
                no_improve_count = 0

                checkpointer.save(agent.state, filename="checkpoint_best")
                print(f"  {Color.BLUE}>> New Best Model Saved{Color.END}")
            else:
                no_improve_count += 1

            # Early stopping
            if target_mean_success is not None:
                if mean_succ >= target_mean_success:
                    success_streak += 1
                else:
                    success_streak = 0

                if success_streak >= patience:
                    print(f"{Color.GREEN}>> Early stopping triggered!{Color.END}")
                    stop_reason = "early_stopping"
                    break

    # Final save
    print(f"{Color.BLUE}>> Saving Final Model...{Color.END}")
    checkpointer.save(agent.state, filename="checkpoint_final")

    stats = {
        "task": task_name,
        "seed": seed,
        "num_envs": num_envs,
        "stop_reason": stop_reason,
        "best_return": best_return,
        "best_success": best_success,
        "wall_time_sec": time.time() - start_time,
        "metrics_history": metrics_history,
    }

    # Close vectorized environment
    vec_env.close()

    return stats


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
    rewind_steps: int = 0,
    resume_from_step: int = 0,
) -> Dict[str, Any]:
    """Core training loop for SAC agent with LTH support (single environment).
    
    Args:
        resume_from_step: If > 0, resume training from this step (checkpoint already loaded).
    """
    start_time = time.time()

    # 0. Setup Logging Files
    os.makedirs(save_dir, exist_ok=True)
    metrics_path = os.path.join(save_dir, "metrics.jsonl")
    episodes_path = os.path.join(save_dir, "train_episodes.jsonl")

    # Only wipe files if starting fresh (not resuming)
    if resume_from_step == 0:
        if os.path.exists(metrics_path):
            os.remove(metrics_path)
        if os.path.exists(episodes_path):
            os.remove(episodes_path)
    else:
        print(f"{Color.BLUE}>> Resuming training from step {resume_from_step}{Color.END}")

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

    # Metric tracking
    metrics_history = []
    train_metrics_buffer = defaultdict(list)

    step = 0
    start_step = resume_from_step + 1 if resume_from_step > 0 else 1

    for step in range(start_step, total_steps + 1):
        if rewind_steps > 0 and step == rewind_steps:
            print(f"{Color.BLUE}>> Saving Rewind Weights (Step {step})...{Color.END}")
            # Don't save optimizer state for rewind - we want fresh optimizers when rewinding
            checkpointer.save(agent.state, filename="checkpoint_rewind", save_optimizer=False)
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
            # Console logging
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

            # Disk logging
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

            # Disk logging
            append_to_jsonl(metrics_path, current_log)

            # Console output
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

                checkpointer.save(agent.state, filename="checkpoint_best")
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

    # Final Save
    print(f"{Color.BLUE}>> Saving Final Model...{Color.END}")
    checkpointer.save(agent.state, filename="checkpoint_final")

    stats = {
        "task": task_name,
        "seed": seed,
        "stop_reason": stop_reason,
        "best_return": best_return,
        "best_success": best_success,
        "wall_time_sec": time.time() - start_time,
        "metrics_history": metrics_history,
    }

    return stats
