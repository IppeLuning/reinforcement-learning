"""Evaluation utilities for RL agents.

This module provides functions for evaluating trained agents,
computing metrics, and tracking performance over time.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol

import numpy as np


class Agent(Protocol):
    """Protocol for agents that can select actions."""
    
    def select_action(self, obs: np.ndarray, eval_mode: bool = False) -> np.ndarray:
        """Select action given observation."""
        ...


def evaluate(
    env: Any,
    agent: Agent,
    num_episodes: int = 5,
    seed: int | None = None,
) -> Dict[str, float]:
    """Evaluate agent performance over multiple episodes.
    
    Runs the agent in evaluation mode (deterministic policy) and collects
    various performance metrics including returns, success rates, and
    action smoothness.
    
    Args:
        env: Gymnasium environment to evaluate in.
        agent: Agent with select_action method.
        num_episodes: Number of episodes to run.
        seed: Optional seed for environment reset.
        
    Returns:
        Dictionary containing:
            - mean_return: Average episodic return.
            - best_return: Best episodic return.
            - mean_success: Average success rate (for Meta-World).
            - best_success: Best success rate.
            - mean_time_to_success: Average steps to first success.
            - mean_smoothness: Average action smoothness (lower is smoother).
    """
    returns: List[float] = []
    successes: List[float] = []
    times_to_success: List[float] = []
    action_smoothness: List[float] = []
    
    for ep_idx in range(num_episodes):
        # Reset environment
        reset_kwargs = {}
        if seed is not None:
            reset_kwargs["seed"] = seed + ep_idx
        
        obs, info = env.reset(**reset_kwargs)
        
        done = False
        truncated = False
        episode_return = 0.0
        episode_success = False
        first_success_step = float("inf")
        episode_actions: List[np.ndarray] = []
        step_count = 0
        
        while not (done or truncated):
            # Get deterministic action
            action = agent.select_action(obs, eval_mode=True)
            episode_actions.append(action)
            
            # Environment step
            obs, reward, done, truncated, info = env.step(action)
            episode_return += reward
            step_count += 1
            
            # Check for success (Meta-World specific)
            success = info.get("success", 0.0)
            if success >= 1.0 and not episode_success:
                episode_success = True
                first_success_step = step_count
        
        # Record episode metrics
        returns.append(episode_return)
        successes.append(1.0 if episode_success else 0.0)
        
        # Time to success (use max steps if no success)
        max_steps = getattr(env, "_max_episode_steps", step_count)
        times_to_success.append(
            first_success_step if episode_success else max_steps
        )
        
        # Action smoothness: mean squared diff between consecutive actions
        if len(episode_actions) > 1:
            actions_arr = np.array(episode_actions)
            diffs = np.diff(actions_arr, axis=0)
            smoothness = np.mean(np.linalg.norm(diffs, axis=1) ** 2)
            action_smoothness.append(smoothness)
    
    # Aggregate metrics
    metrics = {
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
    """Compute Interquartile Mean (IQM) of values.
    
    IQM is a robust measure of central tendency that excludes extreme values.
    It's recommended for RL evaluation (Agarwal et al., 2021).
    
    Args:
        values: List of values to compute IQM over.
        trim_fraction: Fraction of values to trim from each end (default 25%).
        
    Returns:
        Interquartile mean of the values.
    """
    if len(values) == 0:
        return 0.0
    
    sorted_values = np.sort(values)
    n = len(sorted_values)
    
    # Compute indices for IQR
    lower_idx = int(np.floor(n * trim_fraction))
    upper_idx = int(np.ceil(n * (1 - trim_fraction)))
    
    # Handle edge cases
    if lower_idx >= upper_idx:
        return float(np.mean(sorted_values))
    
    return float(np.mean(sorted_values[lower_idx:upper_idx]))


def compute_auc(
    steps: List[int],
    values: List[float],
    normalize: bool = True,
) -> float:
    """Compute Area Under the Curve (AUC) for learning curves.
    
    Uses trapezoidal integration to compute the area under the learning curve.
    Useful for comparing sample efficiency across runs.
    
    Args:
        steps: List of training step numbers.
        values: List of metric values at each step.
        normalize: If True, normalize by total step range.
        
    Returns:
        Area under the curve (optionally normalized).
    """
    if len(steps) < 2:
        return 0.0
    
    # Sort by steps
    sorted_indices = np.argsort(steps)
    sorted_steps = np.array(steps)[sorted_indices]
    sorted_values = np.array(values)[sorted_indices]
    
    # Trapezoidal integration
    auc = np.trapz(sorted_values, sorted_steps)
    
    if normalize:
        step_range = sorted_steps[-1] - sorted_steps[0]
        if step_range > 0:
            auc /= step_range
    
    return float(auc)
