import random

import gymnasium as gym
import metaworld
import numpy as np
from gymnasium.wrappers import TimeLimit, TransformReward


class MetaWorldAdapter(gym.Wrapper):
    """
    Adapts Meta-World to Gymnasium (New API).
    Robustly handles underlying environments that might return
    either 'obs' (Old Gym) or '(obs, info)' (New Gym).
    """

    def __init__(self, env):
        super().__init__(env)
        # Fix action/obs spaces to ensure dtype consistency
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low.astype(np.float32),
            high=env.observation_space.high.astype(np.float32),
            dtype=np.float32,
        )
        self._action_space = env.action_space

    def reset(self, seed=None, options=None):
        # 1. Handle Seeding via Unwrapped (Tunnel to base env)
        if seed is not None:
            if hasattr(self.unwrapped, "seed"):
                self.unwrapped.seed(seed)

        # 2. Call Reset on the raw environment
        ret = self.env.reset()

        # 3. Robustly handle return type (Tuple vs Array)
        if isinstance(ret, tuple):
            # Underlying env is already New-Gym style (obs, info)
            obs, info = ret
        else:
            # Underlying env is Old-Gym style (just obs)
            obs = ret
            info = {}

        return obs.astype(np.float32), info

    def step(self, action):
        # Robustly handle step return
        ret = self.env.step(action)

        if len(ret) == 5:
            # Already Gymnasium style: (obs, reward, term, trunc, info)
            obs, reward, terminated, truncated, info = ret
            return obs.astype(np.float32), float(reward), terminated, truncated, info
        elif len(ret) == 4:
            # Old Gym style: (obs, reward, done, info)
            obs, reward, done, info = ret
            return obs.astype(np.float32), float(reward), False, False, info
        else:
            raise ValueError(f"Unexpected step return length: {len(ret)}")


class MetaWorldTaskSampler(gym.Wrapper):
    """
    Forces Meta-World to sample a new task (goal) on every reset.
    """

    def __init__(self, env, benchmark, mode="train"):
        super().__init__(env)
        self.benchmark = benchmark
        self.mode = mode
        self.task_list = (
            benchmark.train_tasks if mode == "train" else benchmark.test_tasks
        )

    def reset(self, seed=None, options=None):
        # 1. Handle Seeding (Python random + Numpy + Base Env)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            # Use .unwrapped to bypass wrappers and hit the base env
            if hasattr(self.unwrapped, "seed"):
                self.unwrapped.seed(seed)

        # 2. Sample and Set New Goal
        new_task = random.choice(self.task_list)

        # Use .unwrapped to access 'set_task'
        self.unwrapped.set_task(new_task)

        # 3. Standard Reset
        return self.env.reset(seed=seed, options=options)


class SuccessBonusWrapper(gym.Wrapper):
    """
    Adds a sparse bonus when the task is solved.
    """

    def __init__(self, env, bonus=10.0):
        super().__init__(env)
        self.bonus = bonus

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if info.get("success", 0.0) >= 1.0:
            reward += self.bonus

        return obs, reward, terminated, truncated, info


def make_metaworld_env(task_name, max_episode_steps, seed=0):
    """
    Constructs a fully wrapped, SAC-compatible Meta-World environment.
    """
    try:
        ml1 = metaworld.ML1(task_name)
    except KeyError:
        raise ValueError(f"Task '{task_name}' not found in Meta-World ML1.")

    # 1. Create Raw Environment
    env_cls = ml1.train_classes[task_name]
    env = env_cls()

    # 2. Adapt to Gymnasium API (Using the new Robust Adapter)
    env = MetaWorldAdapter(env)

    # 3. Enable Random Goal Sampling
    env = MetaWorldTaskSampler(env, ml1, mode="train")

    # 4. Scale Rewards (Shrink 10,000 -> 10.0)
    env = TransformReward(env, lambda r: r / 10.0)

    # 5. Add Success Bonus (+10.0)
    # env = SuccessBonusWrapper(env, bonus=10.0)

    # 6. Time Limit
    safe_limit = min(max_episode_steps, 500)
    env = TimeLimit(env, max_episode_steps=safe_limit)

    # 7. Seed Immediately
    env.reset(seed=seed)

    # Extract info
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_low = env.action_space.low
    act_high = env.action_space.high

    return env, obs_dim, act_dim, act_low, act_high
