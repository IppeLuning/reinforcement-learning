from __future__ import annotations

import random
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import metaworld
import numpy as np
from gymnasium.wrappers import TimeLimit, TransformReward


class MetaWorldAdapter(gym.Wrapper):
    """Adapts Meta-World to Gymnasium (New API).

    Robustly handles underlying environments that might return
    either 'obs' (Old Gym) or '(obs, info)' (New Gym).

    Attributes:
        observation_space (gym.spaces.Box): The defined observation space.
        _action_space (gym.spaces.Space): The defined action space.
    """

    def __init__(self, env: gym.Env) -> None:
        """Initializes the adapter.

        Args:
            env: The underlying Meta-World environment.
        """
        super().__init__(env)
        self.observation_space: gym.spaces.Box = gym.spaces.Box(
            low=env.observation_space.low.astype(np.float32),
            high=env.observation_space.high.astype(np.float32),
            dtype=np.float32,
        )
        self._action_space: gym.spaces.Space = env.action_space

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment and handles API consistency.

        Args:
            seed: The random seed for initialization.
            options: Additional options for environment reset.

        Returns:
            A tuple of (observation, info).
        """
        if seed is not None:
            if hasattr(self.unwrapped, "seed"):
                self.unwrapped.seed(seed)

        ret = self.env.reset()

        if isinstance(ret, tuple):
            obs, info = ret
        else:
            obs = ret
            info = {}

        return obs.astype(np.float32), info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Applies an action and handles Gymnasium return consistency.

        Args:
            action: The action to perform in the environment.

        Returns:
            A tuple of (observation, reward, terminated, truncated, info).

        Raises:
            ValueError: If the underlying step return length is unsupported.
        """
        ret = self.env.step(action)

        if len(ret) == 5:
            obs, reward, terminated, truncated, info = ret
            return obs.astype(np.float32), float(reward), terminated, truncated, info
        elif len(ret) == 4:
            obs, reward, done, info = ret
            return obs.astype(np.float32), float(reward), False, False, info
        else:
            raise ValueError(f"Unexpected step return length: {len(ret)}")


class MetaWorldTaskSampler(gym.Wrapper):
    """Forces Meta-World to sample a new task (goal) on every reset."""

    def __init__(
        self, env: gym.Env, benchmark: metaworld.Benchmark, mode: str = "train"
    ) -> None:
        """Initializes the task sampler.

        Args:
            env: The environment to wrap.
            benchmark: The Meta-World benchmark instance.
            mode: Either "train" or "test" to sample from respective task sets.
        """
        super().__init__(env)
        self.benchmark: metaworld.Benchmark = benchmark
        self.mode: str = mode
        self.task_list: list = (
            benchmark.train_tasks if mode == "train" else benchmark.test_tasks
        )

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Samples a new task and resets the environment.

        Args:
            seed: The random seed.
            options: Reset options.

        Returns:
            The initial observation and info.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            if hasattr(self.unwrapped, "seed"):
                self.unwrapped.seed(seed)

        new_task = random.choice(self.task_list)
        self.unwrapped.set_task(new_task)

        return self.env.reset(seed=seed, options=options)


class SuccessBonusWrapper(gym.Wrapper):
    """Adds a sparse bonus when the task is solved."""

    def __init__(self, env: gym.Env, bonus: float = 10.0) -> None:
        """Initializes the success bonus wrapper.

        Args:
            env: The environment to wrap.
            bonus: The amount of reward to add upon success.
        """
        super().__init__(env)
        self.bonus: float = bonus

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Checks for success and applies reward bonus.

        Args:
            action: The action to perform.

        Returns:
            The standard Gymnasium step tuple.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)

        if info.get("success", 0.0) >= 1.0:
            reward += self.bonus

        return obs, reward, terminated, truncated, info


def make_metaworld_env(
    task_name: str,
    max_episode_steps: int,
    scale_factor: float,
    seed: int = 0,
) -> Tuple[gym.Env, int, int, np.ndarray, np.ndarray]:
    """Constructs a fully wrapped, SAC-compatible Meta-World environment.

    Args:
        task_name: Name of the ML1 task (e.g., 'reach-v2').
        max_episode_steps: Maximum number of steps allowed per episode.
        scale_factor: Factor by which to divide the raw reward.
        seed: Random seed for initialization.

    Returns:
        A tuple containing:
            - The wrapped Gymnasium environment.
            - Observation space dimension.
            - Action space dimension.
            - Action space lower bounds.
            - Action space upper bounds.

    Raises:
        ValueError: If the specified task_name does not exist in ML1.
    """
    try:
        ml1 = metaworld.ML1(task_name)
    except KeyError:
        raise ValueError(f"Task '{task_name}' not found in Meta-World ML1.")

    env_cls = ml1.train_classes[task_name]
    env = env_cls()

    env = MetaWorldAdapter(env)
    env = MetaWorldTaskSampler(env, ml1, mode="train")
    env = TransformReward(env, lambda r: r / scale_factor)

    safe_limit = min(max_episode_steps, 500)
    env = TimeLimit(env, max_episode_steps=safe_limit)

    env.reset(seed=seed)

    obs_dim = int(env.observation_space.shape[0])
    act_dim = int(env.action_space.shape[0])
    act_low = env.action_space.low
    act_high = env.action_space.high

    return env, obs_dim, act_dim, act_low, act_high
