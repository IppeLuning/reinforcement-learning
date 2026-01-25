from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.training.evaluation import evaluate

if TYPE_CHECKING:
    from src.agents.sac import SACAgent
    from src.data.replay_buffer import ReplayBuffer
    from src.utils.checkpointing import Checkpointer


class Color:
    """ANSI color codes for formatted terminal output."""

    GREEN: str = "\033[92m"
    RED: str = "\033[91m"
    YELLOW: str = "\033[93m"
    BLUE: str = "\033[94m"
    END: str = "\033[0m"


def append_to_jsonl(path: str, data: Dict[str, Any]) -> None:
    """Appends a dictionary as a single JSON line to a specified file.

    Ensures that numeric types (like NumPy floats/ints) are converted to
    native Python types for JSON serialization.

    Args:
        path: Path to the .jsonl file.
        data: Dictionary of metrics to log.
    """
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
    vec_env: Any,
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
) -> Dict[str, Any]:
    """Executes a training loop optimized for vectorized environments.

    This loop collects data from multiple environments in parallel, which is
    essential for speeding up the data collection bottleneck in SAC.

    Args:
        vec_env: A vectorized environment (e.g., VectorizedMetaWorldEnv).
        agent: The SAC agent instance.
        replay_buffer: Buffer for storing transitions.
        total_steps: Total environment steps to perform across all envs.
        start_steps: Number of steps to collect with random actions initially.
        batch_size: Size of training batches.
        eval_interval: Step frequency for periodic evaluation.
        save_dir: Directory to save metrics and checkpoints.
        seed: Random seed for reproducibility.
        task_name: Name of the Meta-World task.
        num_envs: Number of parallel environments in the vectorized wrapper.
        scale_factor: Reward scaling factor.
        target_mean_success: Optional success rate threshold for early stopping.
        patience: Number of evaluations to wait for improvement before stopping.
        updates_per_step: Ratio of gradient updates per environment step.
        eval_episodes: Number of episodes to run during evaluation.
        checkpointer: Utility for saving/restoring model states.
        rewind_steps: Global step at which to save "rewind" weights for LTH.
        max_episode_steps: Maximum steps allowed per episode.

    Returns:
        A dictionary containing final training statistics and history.
    """
    start_time = time.time()

    os.makedirs(save_dir, exist_ok=True)
    metrics_path = os.path.join(save_dir, "metrics.jsonl")
    episodes_path = os.path.join(save_dir, "train_episodes.jsonl")

    if os.path.exists(metrics_path):
        os.remove(metrics_path)
    if os.path.exists(episodes_path):
        os.remove(episodes_path)

    obs, _ = vec_env.reset(seed=seed)

    episode_returns = np.zeros(num_envs)
    episode_successes = np.zeros(num_envs)
    episode_lens = np.zeros(num_envs, dtype=int)

    best_return = float("-inf")
    best_success = 0.0

    recent_successes: List[float] = []
    recent_returns: List[float] = []

    success_streak = 0
    stop_reason = "max_steps"

    metrics_history: List[Dict[str, Any]] = []
    train_metrics_buffer: Dict[str, List[float]] = defaultdict(list)

    total_env_steps = num_envs
    rewind_saved = False

    print(f"{Color.BLUE}>> Training with {num_envs} parallel environments{Color.END}")

    while total_env_steps < total_steps:
        # 1. Action Selection
        if total_env_steps < start_steps:
            actions = np.array([vec_env.action_space.sample() for _ in range(num_envs)])
        else:
            actions = np.array(
                [agent.select_action(obs[i], eval_mode=False) for i in range(num_envs)]
            )

        # 2. Parallel Step
        next_obs, rewards, dones, truncateds, infos = vec_env.step(actions)
        terminals = np.logical_or(dones, truncateds)

        # 3. Store and Track
        for i in range(num_envs):
            replay_buffer.store(
                obs[i], actions[i], rewards[i], next_obs[i], float(dones[i])
            )

            episode_returns[i] += rewards[i]
            episode_lens[i] += 1

            try:
                if isinstance(infos, dict):
                    if "final_info" in infos and i < len(infos["final_info"]):
                        final_info = infos["final_info"][i]
                        if final_info is not None and "success" in final_info:
                            episode_successes[i] = max(
                                episode_successes[i], float(final_info["success"])
                            )
                    elif "success" in infos and hasattr(
                        infos["success"], "__getitem__"
                    ):
                        episode_successes[i] = max(
                            episode_successes[i], float(infos["success"][i])
                        )
                elif isinstance(infos, (list, tuple)) and i < len(infos):
                    if isinstance(infos[i], dict) and "success" in infos[i]:
                        episode_successes[i] = max(
                            episode_successes[i], float(infos[i]["success"])
                        )
            except (KeyError, IndexError, TypeError):
                pass

        total_env_steps += num_envs
        obs = next_obs

        if rewind_steps > 0 and not rewind_saved and total_env_steps >= rewind_steps:
            if checkpointer:
                print(
                    f"{Color.BLUE}>> Saving Rewind Weights (Step {total_env_steps})...{Color.END}"
                )
                checkpointer.save(agent.state, filename="checkpoint_rewind")
                rewind_saved = True

        # 4. Handle episode endings
        for i in range(num_envs):
            if terminals[i]:
                recent_successes.append(float(episode_successes[i]))
                recent_returns.append(float(episode_returns[i]))
                if len(recent_successes) > 100:
                    recent_successes.pop(0)
                    recent_returns.pop(0)

                color = Color.GREEN if episode_successes[i] >= 1.0 else Color.RED
                print(
                    f"Step {total_env_steps} | Env {i} | Return: {episode_returns[i]:.1f} | "
                    f"{color}Success: {episode_successes[i]:.2f}{Color.END} | "
                    f"Avg Succ (100): {np.mean(recent_successes):.2f} | "
                    f"Elapsed: {(time.time() - start_time)/60:.1f}min"
                )

                episode_log = {
                    "step": total_env_steps,
                    "env_id": i,
                    "episode_return": float(episode_returns[i]),
                    "episode_success": float(episode_successes[i]),
                    "episode_length": int(episode_lens[i]),
                    "wall_time": time.time() - start_time,
                }
                append_to_jsonl(episodes_path, episode_log)

                episode_returns[i] = 0.0
                episode_successes[i] = 0.0
                episode_lens[i] = 0

        # 5. Training Updates
        if total_env_steps >= start_steps and replay_buffer.is_ready(batch_size):
            num_updates = int(updates_per_step * num_envs)
            for _ in range(num_updates):
                metrics = agent.update(replay_buffer, batch_size)
                for k, v in metrics.items():
                    train_metrics_buffer[k].append(float(v))

        # 6. Evaluation & Logging
        if (
            total_env_steps >= eval_interval
            and total_env_steps % eval_interval < num_envs
        ):
            from src.envs.factory import make_metaworld_env

            eval_env, _, _, _, _ = make_metaworld_env(
                task_name, max_episode_steps, scale_factor, seed=seed
            )

            eval_metrics = evaluate(eval_env, agent, num_episodes=eval_episodes)
            eval_env.close()

            avg_train_metrics = {
                f"train/{k}": np.mean(v) if v else 0.0
                for k, v in train_metrics_buffer.items()
            }
            train_metrics_buffer = defaultdict(list)

            current_log = {
                "step": total_env_steps,
                "wall_time": time.time() - start_time,
                **eval_metrics,
                **avg_train_metrics,
            }
            metrics_history.append(current_log)
            append_to_jsonl(metrics_path, current_log)

            mean_ret = eval_metrics["mean_return"]
            mean_succ = eval_metrics["mean_success"]

            print(
                f"[Eval] Step {total_env_steps:>7} | "
                f"Ret: {mean_ret:>6.1f} | "
                f"Succ: {mean_succ:>4.2f} | "
                f"Alpha: {agent.alpha:.3f}"
            )

            if checkpointer:
                checkpointer.save(
                    agent.state, filename=f"checkpoint_step_{total_env_steps}"
                )

                if mean_ret > best_return:
                    best_return = mean_ret
                    best_success = mean_succ
                    checkpointer.save(agent.state, filename="checkpoint_best")
                    print(f"  {Color.BLUE}>> New Best Model Saved{Color.END}")

            if target_mean_success is not None:
                if mean_succ >= target_mean_success:
                    success_streak += 1
                else:
                    success_streak = 0

                if success_streak >= patience:
                    print(f"{Color.GREEN}>> Early stopping triggered!{Color.END}")
                    stop_reason = "early_stopping"
                    break

    if checkpointer:
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
) -> Dict[str, Any]:
    """Core training loop for SAC agent in a single environment.

    Args:
        env: Single gymnasium environment.
        agent: The SAC agent instance.
        replay_buffer: Buffer for storing transitions.
        total_steps: Total number of environment steps.
        start_steps: Initial random exploration steps.
        batch_size: Training batch size.
        eval_interval: Periodic evaluation frequency.
        save_dir: Logging and checkpoint directory.
        seed: Random seed.
        task_name: Environment task name.
        target_mean_success: Optional threshold for early stopping.
        patience: Evaluations to wait for success streak.
        updates_per_step: Gradient updates per step.
        eval_episodes: Evaluation episode count.
        checkpointer: Utility for model saving.
        rewind_steps: Step at which to save rewind weights.

    Returns:
        Final training statistics and history.
    """
    start_time = time.time()

    os.makedirs(save_dir, exist_ok=True)
    metrics_path = os.path.join(save_dir, "metrics.jsonl")
    episodes_path = os.path.join(save_dir, "train_episodes.jsonl")

    if os.path.exists(metrics_path):
        os.remove(metrics_path)
    if os.path.exists(episodes_path):
        os.remove(episodes_path)

    obs, _ = env.reset(seed=seed)

    episode_return = 0.0
    episode_success = 0.0
    episode_len = 0

    best_return = float("-inf")
    best_success = 0.0

    recent_successes: List[float] = []
    recent_returns: List[float] = []

    success_streak = 0
    stop_reason = "max_steps"

    metrics_history: List[Dict[str, Any]] = []
    train_metrics_buffer: Dict[str, List[float]] = defaultdict(list)

    for step in range(1, total_steps + 1):
        if rewind_steps > 0 and step == rewind_steps and checkpointer:
            print(f"{Color.BLUE}>> Saving Rewind Weights (Step {step})...{Color.END}")
            checkpointer.save(agent.state, filename="checkpoint_rewind")

        # 1. Action Selection
        if step < start_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, eval_mode=False)

        # 2. Step
        next_obs, reward, done, truncated, info = env.step(action)
        terminal = done or truncated

        # 3. Store
        replay_buffer.store(obs, action, reward, next_obs, float(done))

        obs = next_obs
        episode_return += float(reward)
        episode_success = max(episode_success, float(info.get("success", 0.0)))
        episode_len += 1

        # 4. Handle Episode End
        if terminal:
            recent_successes.append(episode_success)
            recent_returns.append(episode_return)
            if len(recent_successes) > 100:
                recent_successes.pop(0)

            color = Color.GREEN if episode_success >= 1.0 else Color.RED
            if episode_success >= 1.0:
                print(
                    f"Step {step} | Return: {episode_return:.1f} | "
                    f"{color}Success: {episode_success}{Color.END} | "
                    f"Elapsed: {(time.time() - start_time)/60:.1f}min"
                )

            episode_log = {
                "step": step,
                "episode_return": episode_return,
                "episode_success": episode_success,
                "episode_length": episode_len,
                "wall_time": time.time() - start_time,
            }
            append_to_jsonl(episodes_path, episode_log)

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
            eval_metrics = evaluate(env, agent, num_episodes=eval_episodes)

            avg_train_metrics = {
                f"train/{k}": np.mean(v) if v else 0.0
                for k, v in train_metrics_buffer.items()
            }
            train_metrics_buffer = defaultdict(list)

            current_log = {
                "step": step,
                "wall_time": time.time() - start_time,
                **eval_metrics,
                **avg_train_metrics,
            }
            metrics_history.append(current_log)
            append_to_jsonl(metrics_path, current_log)

            mean_ret = eval_metrics["mean_return"]
            mean_succ = eval_metrics["mean_success"]

            print(
                f"[Eval] Step {step:>7} | Ret: {mean_ret:>6.1f} | Succ: {mean_succ:>4.2f}"
            )

            if checkpointer:
                if mean_ret > best_return:
                    best_return = mean_ret
                    best_success = mean_succ
                    checkpointer.save(agent.state, filename="checkpoint_best")
                    print(f"  {Color.BLUE}>> New Best Model Saved{Color.END}")

            if target_mean_success is not None:
                if mean_succ >= target_mean_success:
                    success_streak += 1
                else:
                    success_streak = 0

                if success_streak >= patience:
                    print(f"{Color.GREEN}>> Early stopping triggered!{Color.END}")
                    stop_reason = "early_stopping"
                    break

    if checkpointer:
        print(f"{Color.BLUE}>> Saving Final Model...{Color.END}")
        checkpointer.save(agent.state, filename="checkpoint_final")

    return {
        "task": task_name,
        "seed": seed,
        "stop_reason": stop_reason,
        "best_return": best_return,
        "best_success": best_success,
        "wall_time_sec": time.time() - start_time,
        "metrics_history": metrics_history,
    }
