from typing import Dict

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MultiTaskEnv(gym.Env):
    def __init__(self, task_envs: Dict[str, gym.Env], max_episode_steps: int = 200):
        super().__init__()
        self.task_envs = task_envs
        self.task_names = list(task_envs.keys())
        self.num_tasks = len(self.task_names)
        self.max_episode_steps = max_episode_steps

        # Assume all envs have compatible spaces (Standard in Meta-World ML1)
        base_env = task_envs[self.task_names[0]]
        self.action_space = base_env.action_space

        # Augment obs: Original + One-Hot Task ID
        original_obs_dim = base_env.observation_space.shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(original_obs_dim + self.num_tasks,),
            dtype=np.float32,
        )

        self.active_task_idx = 0
        self.active_env = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        # Sample a new task
        self.active_task_idx = np.random.randint(self.num_tasks)
        task_name = self.task_names[self.active_task_idx]
        self.active_env = self.task_envs[task_name]

        obs, info = self.active_env.reset(seed=seed)
        info["task_name"] = task_name

        return self._augment_obs(obs, self.active_task_idx), info

    def step(self, action):
        obs, reward, done, truncated, info = self.active_env.step(action)
        info["task_name"] = self.task_names[self.active_task_idx]
        return (
            self._augment_obs(obs, self.active_task_idx),
            reward,
            done,
            truncated,
            info,
        )

    def _augment_obs(self, obs, task_idx):
        one_hot = np.zeros(self.num_tasks, dtype=np.float32)
        one_hot[task_idx] = 1.0
        return np.concatenate([obs, one_hot], axis=-1)

    def close(self):
        for env in self.task_envs.values():
            env.close()
