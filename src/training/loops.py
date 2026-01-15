from __future__ import annotations

import time
from collections import defaultdict
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

    # Reset environment
    obs, _ = env.reset(seed=seed)

    # Episode tracking
    episode_return = 0.0
    episode_success = 0.0
    episode_len = 0

    # Best performance tracking
    best_return = float("-inf")
    best_success = 0.0
    all_results = []
    step = 0

    # Early stopping tracking
    no_improve_count = 0
    success_streak = 0
    stop_reason = "max_steps"

    # Metric tracking
    metrics_history = []
    # Accumulate training losses between evaluations
    train_metrics_buffer = defaultdict(list)

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
        # Note: We store 'done' (termination), not 'truncated' (time limit)
        # for correct bootstrapping.
        replay_buffer.store(obs, action, reward, next_obs, float(done))

        # Update tracking
        obs = next_obs
        episode_return += reward
        episode_success = max(episode_success, float(info.get("success", 0.0)))
        episode_len += 1

        # 4. Handle Episode End
        if terminal:
            color = Color.GREEN if episode_success >= 1.0 else Color.RED
            all_results.append(episode_success)
            # Optional: Only print every few episodes to reduce clutter
            if episode_success >= 1.0:
                print(
                    f"Step {step} | Return: {episode_return:.1f} | {color}Success: {episode_success}{Color.END} | Average Success (last 100): {np.mean(all_results[-100:]):.2f}"
                )

            obs, _ = env.reset()
            episode_return = 0.0
            episode_success = 0.0
            episode_len = 0

        # 5. Training Updates
        if step >= start_steps and replay_buffer.is_ready(batch_size):
            for _ in range(updates_per_step):
                # Now capturing the metrics!
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
            # Clear buffer
            train_metrics_buffer = defaultdict(list)

            # Combine all metrics
            current_log = {
                "step": step,
                "wall_time": time.time() - start_time,
                **eval_metrics,
                **avg_train_metrics,
            }
            metrics_history.append(current_log)

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
                best_step = step
                no_improve_count = 0

                checkpointer.save(
                    agent.state,
                    filename="checkpoint_best",  # Explicit name is safer
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

    # Final Stats
    total_time = time.time() - start_time
    stats = {
        "task": task_name,
        "seed": seed,
        "stop_reason": stop_reason,
        "best_return": best_return,
        "best_success": best_success,
        "wall_time_sec": total_time,
        "metrics_history": metrics_history,
    }

    return stats
