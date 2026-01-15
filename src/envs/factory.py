import random

import gymnasium as gym
import metaworld
import numpy as np
from gymnasium.wrappers import TimeLimit


class MetaWorldTaskSampler(gym.Wrapper):
    """
    Forces Meta-World to sample a new task (goal) on every reset.
    Without this, env.reset() just restarts the SAME task/goal repeatedly.
    """

    def __init__(self, env, benchmark, mode="train"):
        super().__init__(env)
        self.benchmark = benchmark
        self.mode = mode
        # ML1 provides 'train_tasks' (50 goals) and 'test_tasks' (holdout goals)
        self.task_list = (
            benchmark.train_tasks if mode == "train" else benchmark.test_tasks
        )

    def reset(self, seed=None, options=None):
        # 1. Seed the RNG for reproducibility (critical for 'seed' argument)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # 2. Sample a new task (Goal) from the list
        # This is the magic line that moves the ball/goal
        new_task = random.choice(self.task_list)
        self.env.set_task(new_task)

        # 3. Reset the physics
        return self.env.reset(seed=seed, options=options)


class SuccessBonusWrapper(gym.Wrapper):
    """
    Adds a massive bonus to the reward when the task is successfully completed.
    Helps the agent distinguish between 'hovering near' and 'actually winning'.
    """

    def __init__(self, env, bonus=10.0):
        super().__init__(env)
        self.bonus = bonus

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        # Check if Meta-World thinks we succeeded
        if info.get("success", 0.0) >= 1.0:
            reward += self.bonus

        return obs, reward, done, truncated, info


class ScaleRewardWrapper(gym.RewardWrapper):
    def __init__(self, env, scale=0.01):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward):
        return reward * self.scale


def make_metaworld_env(task_name, max_episode_steps, seed=None):
    # 1. Instantiate ML1 to get the benchmark and task list
    # ML1 contains the distribution of goals for this specific task
    ml1 = metaworld.ML1(task_name)

    # 2. Create the raw environment
    env_cls = ml1.train_classes[task_name]
    env = env_cls()

    # 3. Apply the Task Sampler Wrapper (The Fix)
    env = MetaWorldTaskSampler(env, ml1, mode="train")

    env = SuccessBonusWrapper(env, bonus=20.0)

    # 4. Apply TimeLimit (Standard Gym wrapper)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    # Optional: Seed immediately if provided
    if seed is not None:
        env.reset(seed=seed)

    # Extract dimensions for your agent
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_low = env.action_space.low
    act_high = env.action_space.high

    return env, obs_dim, act_dim, act_low, act_high
