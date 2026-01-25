from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Union

import numpy as np


class Agent(Protocol):
    """Protocol for agents that can select actions.

    This ensures that any object passed to the evaluate function implements
     the required action selection interface.
    """

    def select_action(self, obs: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        """Selects an action given an observation.

        Args:
            obs: The current environment observation.
            eval_mode: Whether to use a deterministic policy for evaluation.

        Returns:
            The selected action as a NumPy array.
        """
        ...


def evaluate(
    env: Any,
    agent: Agent,
    num_episodes: int = 5,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluates agent performance over multiple episodes.

    Runs the agent in evaluation mode (deterministic policy) and collects
    various performance metrics including returns, success rates, and
    action smoothness.

    Args:
        env: Gymnasium environment to evaluate in.
        agent: Agent implementing the select_action method.
        num_episodes: Number of episodes to run for the evaluation.
        seed: Optional seed for environment reset to ensure reproducibility.

    Returns:
        A dictionary containing the following aggregated metrics:
            - mean_return: Average episodic return.
            - best_return: Best episodic return recorded.
            - std_return: Standard deviation of returns.
            - mean_success: Average success rate (specifically for Meta-World).
            - best_success: Best success rate (1.0 or 0.0).
            - mean_time_to_success: Average steps taken to reach first success.
            - mean_smoothness: Average action smoothness (mean squared difference).
    """
    returns: List[float] = []
    successes: List[float] = []
    times_to_success: List[float] = []
    action_smoothness: List[float] = []

    for ep_idx in range(num_episodes):
        reset_kwargs: Dict[str, Any] = {}
        if seed is not None:
            reset_kwargs["seed"] = seed + ep_idx

        obs, info = env.reset(**reset_kwargs)

        done: bool = False
        truncated: bool = False
        episode_return: float = 0.0
        episode_success: bool = False
        first_success_step: float = float("inf")
        episode_actions: List[np.ndarray] = []
        step_count: int = 0

        while not (done or truncated):
            action = agent.select_action(obs, eval_mode=True)
            episode_actions.append(action)

            obs, reward, done, truncated, info = env.step(action)
            episode_return += float(reward)
            step_count += 1

            success = info.get("success", 0.0)
            if success >= 1.0 and not episode_success:
                episode_success = True
                first_success_step = float(step_count)

        returns.append(episode_return)
        successes.append(1.0 if episode_success else 0.0)

        max_steps = getattr(env, "_max_episode_steps", step_count)
        times_to_success.append(
            first_success_step if episode_success else float(max_steps)
        )

        if len(episode_actions) > 1:
            actions_arr = np.array(episode_actions)
            diffs = np.diff(actions_arr, axis=0)
            smoothness = np.mean(np.linalg.norm(diffs, axis=1) ** 2)
            action_smoothness.append(float(smoothness))

    metrics: Dict[str, float] = {
        "mean_return": float(np.mean(returns)),
        "best_return": float(np.max(returns)),
        "std_return": float(np.std(returns)),
        "mean_success": float(np.mean(successes)),
        "best_success": float(np.max(successes)),
        "mean_time_to_success": float(np.mean(times_to_success)),
        "mean_smoothness": (
            float(np.mean(action_smoothness)) if action_smoothness else 0.0
        ),
    }

    return metrics


def compute_iqm(values: List[float], trim_fraction: float = 0.25) -> float:
    """Computes the Interquartile Mean (IQM) of a list of values.

    IQM is a robust measure of central tendency that excludes extreme values,
    making it resilient to outliers in RL experiments.

    Args:
        values: List of numeric values to compute IQM over.
        trim_fraction: Fraction of values to trim from each end (default 25%).

    Returns:
        The interquartile mean of the values.
    """
    if not values:
        return 0.0

    sorted_values = np.sort(values)
    n = len(sorted_values)

    lower_idx = int(np.floor(n * trim_fraction))
    upper_idx = int(np.ceil(n * (1 - trim_fraction)))

    if lower_idx >= upper_idx:
        return float(np.mean(sorted_values))

    return float(np.mean(sorted_values[lower_idx:upper_idx]))


def compute_auc(
    steps: List[int],
    values: List[float],
    normalize: bool = True,
) -> float:
    """Computes the Area Under the Curve (AUC) for learning curves.

    Uses trapezoidal integration to compute the area, which serves as a metric
    for aggregate performance and sample efficiency over time.

    Args:
        steps: List of training step numbers (x-axis).
        values: List of metric values at each step (y-axis).
        normalize: If True, divides the AUC by the total step range.

    Returns:
        The calculated area under the curve.
    """
    if len(steps) < 2:
        return 0.0

    sorted_indices = np.argsort(steps)
    sorted_steps = np.array(steps)[sorted_indices]
    sorted_values = np.array(values)[sorted_indices]

    auc = float(np.trapz(sorted_values, sorted_steps))

    if normalize:
        step_range = float(sorted_steps[-1] - sorted_steps[0])
        if step_range > 0:
            auc /= step_range

    return auc
