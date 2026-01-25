from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv


class VectorizedMetaWorldEnv:
    """Wrapper for running multiple Meta-World environments in parallel.

    Supports two strategies:
    - 'sync': All environments step together in the same process (CPU efficient)
    - 'async': Each environment runs in a separate process (prevents bottlenecks)

    This dramatically speeds up data collection compared to stepping one env at a time.

    Attributes:
        num_envs (int): Number of parallel environments.
        strategy (str): The vectorization strategy used ('sync' or 'async').
        envs (Union[SyncVectorEnv, AsyncVectorEnv]): The underlying Gymnasium vector environment.
        observation_space (gym.spaces.Space): The observation space for a single environment.
        action_space (gym.spaces.Space): The action space for a single environment.
    """

    def __init__(
        self,
        env_fns: List[Callable[[], gym.Env]],
        strategy: str = "sync",
    ) -> None:
        """Initializes the vectorized environment.

        Args:
            env_fns: List of callables that create environments when called.
            strategy: Either 'sync' or 'async' for vectorization strategy.

        Raises:
            ValueError: If an unknown strategy is provided.
        """
        self.num_envs: int = len(env_fns)
        self.strategy: str = strategy

        if strategy == "sync":
            self.envs: SyncVectorEnv = SyncVectorEnv(env_fns)
        elif strategy == "async":
            self.envs: AsyncVectorEnv = AsyncVectorEnv(env_fns)
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'sync' or 'async'")

        self.observation_space: gym.spaces.Space = self.envs.single_observation_space
        self.action_space: gym.spaces.Space = self.envs.single_action_space

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets all environments in parallel.

        Args:
            seed: Optional base seed. If provided, each sub-environment receives seed + i.

        Returns:
            A tuple containing the batched initial observations and the info dictionary.
        """
        if seed is not None:
            seeds = [seed + i for i in range(self.num_envs)]
            obs, infos = self.envs.reset(seed=seeds)
        else:
            obs, infos = self.envs.reset()
        return obs, infos

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Steps all environments in parallel.

        Args:
            actions: Array of shape (num_envs, action_dim).

        Returns:
            A tuple containing:
                - obs: Array of shape (num_envs, obs_dim).
                - rewards: Array of shape (num_envs,).
                - terminateds: Array of shape (num_envs,).
                - truncateds: Array of shape (num_envs,).
                - infos: Dict with keys containing arrays of shape (num_envs,).
        """
        return self.envs.step(actions)

    def close(self) -> None:
        """Closes all environments and shuts down processes if async."""
        self.envs.close()

    def __del__(self) -> None:
        """Ensures environments are closed on object deletion."""
        if hasattr(self, "envs"):
            self.envs.close()


def make_vectorized_metaworld_env(
    task_name: str,
    max_episode_steps: int,
    scale_factor: float,
    num_envs: int = 4,
    strategy: str = "sync",
    base_seed: int = 0,
) -> Tuple[VectorizedMetaWorldEnv, int, int, np.ndarray, np.ndarray]:
    """Creates a vectorized Meta-World environment.

    Args:
        task_name: Name of the Meta-World task (e.g., 'reach-v3').
        max_episode_steps: Maximum steps per episode.
        scale_factor: Factor by which to divide the raw reward.
        num_envs: Number of parallel environments.
        strategy: 'sync' or 'async' vectorization.
        base_seed: Base random seed (each env gets base_seed + i).

    Returns:
        A tuple containing:
            - vec_env: Vectorized environment instance.
            - obs_dim: Observation dimension.
            - act_dim: Action dimension.
            - act_low: Action space lower bounds.
            - act_high: Action space upper bounds.
    """
    from src.envs.factory import make_metaworld_env

    env_fns = [
        lambda i=i: make_metaworld_env(
            task_name, max_episode_steps - 1, scale_factor, seed=base_seed + i
        )[0]
        for i in range(num_envs)
    ]

    vec_env = VectorizedMetaWorldEnv(env_fns, strategy=strategy)

    single_env, obs_dim, act_dim, act_low, act_high = make_metaworld_env(
        task_name, max_episode_steps - 1, scale_factor, seed=base_seed
    )
    single_env.close()

    return vec_env, obs_dim, act_dim, act_low, act_high
