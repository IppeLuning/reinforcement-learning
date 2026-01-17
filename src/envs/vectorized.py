"""Vectorized environment wrappers for parallel data collection."""

from __future__ import annotations

from typing import Any, Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from matplotlib.pyplot import sca


class VectorizedMetaWorldEnv:
    """
    Wrapper for running multiple Meta-World environments in parallel.

    Supports two strategies:
    - 'sync': All environments step together in the same process (CPU efficient)
    - 'async': Each environment runs in a separate process (prevents bottlenecks)

    This dramatically speeds up data collection compared to stepping one env at a time.
    """

    def __init__(
        self,
        env_fns: list[Callable[[], gym.Env]],
        strategy: str = "sync",
    ):
        """
        Initialize vectorized environment.

        Args:
            env_fns: List of callables that create environments when called
            strategy: Either 'sync' or 'async' for vectorization strategy
        """
        self.num_envs = len(env_fns)
        self.strategy = strategy

        if strategy == "sync":
            self.envs = SyncVectorEnv(env_fns)
        elif strategy == "async":
            self.envs = AsyncVectorEnv(env_fns)
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'sync' or 'async'")

        # Expose standard gym attributes
        self.observation_space = self.envs.single_observation_space
        self.action_space = self.envs.single_action_space

    def reset(self, seed: Optional[int] = None):
        """Reset all environments in parallel."""
        if seed is not None:
            # Seed each env with a different seed for diversity
            seeds = [seed + i for i in range(self.num_envs)]
            obs, infos = self.envs.reset(seed=seeds)
        else:
            obs, infos = self.envs.reset()
        return obs, infos

    def step(self, actions: np.ndarray):
        """
        Step all environments in parallel.

        Args:
            actions: Array of shape (num_envs, action_dim)

        Returns:
            obs: Array of shape (num_envs, obs_dim)
            rewards: Array of shape (num_envs,)
            dones: Array of shape (num_envs,)
            truncateds: Array of shape (num_envs,)
            infos: Dict with keys like 'success' containing arrays of shape (num_envs,)
        """
        return self.envs.step(actions)

    def close(self):
        """Close all environments."""
        self.envs.close()

    def __del__(self):
        """Ensure environments are closed on deletion."""
        if hasattr(self, "envs"):
            self.envs.close()


def make_vectorized_metaworld_env(
    task_name: str,
    max_episode_steps: int,
    scale_factor: float,
    num_envs: int = 4,
    strategy: str = "sync",
    base_seed: int = 0,
) -> tuple[VectorizedMetaWorldEnv, int, int, np.ndarray, np.ndarray]:
    """
    Create a vectorized Meta-World environment.

    Args:
        task_name: Name of the Meta-World task (e.g., 'reach-v3')
        max_episode_steps: Maximum steps per episode
        num_envs: Number of parallel environments
        strategy: 'sync' or 'async' vectorization
        base_seed: Base random seed (each env gets base_seed + i)

    Returns:
        vec_env: Vectorized environment
        obs_dim: Observation dimension
        act_dim: Action dimension
        act_low: Action space lower bounds
        act_high: Action space upper bounds
    """
    from src.envs.factory import make_metaworld_env

    # Create environment factory functions
    # Each env needs a unique seed for diversity
    env_fns = [
        lambda i=i: make_metaworld_env(
            task_name, max_episode_steps, scale_factor, seed=base_seed + i
        )[
            0
        ]  # [0] to get just the env, not the metadata
        for i in range(num_envs)
    ]

    # Create vectorized environment
    vec_env = VectorizedMetaWorldEnv(env_fns, strategy=strategy)

    # Get environment metadata from the first environment
    single_env, obs_dim, act_dim, act_low, act_high = make_metaworld_env(
        task_name, max_episode_steps, scale_factor, seed=base_seed
    )
    single_env.close()

    return vec_env, obs_dim, act_dim, act_low, act_high
