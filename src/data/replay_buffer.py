from __future__ import annotations

from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np

from src.utils.types import Batch


class ReplayBuffer:
    """Experience replay buffer for off-policy RL algorithms.

    Stores transitions (obs, action, reward, next_obs, done) in a circular buffer
    using NumPy arrays for memory efficiency. Sampling returns JAX arrays
    ready for training.

    Note: This buffer stores data on CPU (NumPy) and converts to JAX arrays
    only when sampling. This is more memory-efficient for large buffers.

    Attributes:
        obs_dim (int): Dimension of observations.
        act_dim (int): Dimension of actions.
        max_size (int): Maximum number of transitions to store.
        obs_buf (np.ndarray): Buffer for observations.
        next_obs_buf (np.ndarray): Buffer for next observations.
        action_buf (np.ndarray): Buffer for actions.
        reward_buf (np.ndarray): Buffer for rewards.
        done_buf (np.ndarray): Buffer for terminal flags.
        ptr (int): Current write position in the circular buffer.
        size (int): Current number of transitions stored.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        max_size: int = 1_000_000,
    ) -> None:
        """Initializes the replay buffer.

        Args:
            obs_dim: Dimension of observations.
            act_dim: Dimension of actions.
            max_size: Maximum number of transitions to store.
        """
        self.obs_dim: int = obs_dim
        self.act_dim: int = act_dim
        self.max_size: int = max_size

        self.obs_buf: np.ndarray = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.next_obs_buf: np.ndarray = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.action_buf: np.ndarray = np.zeros((max_size, act_dim), dtype=np.float32)
        self.reward_buf: np.ndarray = np.zeros((max_size, 1), dtype=np.float32)
        self.done_buf: np.ndarray = np.zeros((max_size, 1), dtype=np.float32)

        self.ptr: int = 0
        self.size: int = 0

    def store(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: float,
    ) -> None:
        """Stores a single transition in the buffer.

        Args:
            obs: Observation, shape (obs_dim,).
            action: Action taken, shape (act_dim,).
            reward: Reward received (scalar).
            next_obs: Next observation, shape (obs_dim,).
            done: Whether episode terminated (0.0 or 1.0).
        """
        self.obs_buf[self.ptr] = obs
        self.action_buf[self.ptr] = action
        self.reward_buf[self.ptr] = reward
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(
        self,
        batch_size: int,
        key: Optional[jax.Array] = None,
    ) -> Batch:
        """Samples a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample.
            key: Optional JAX random key. If None, uses NumPy random.

        Returns:
            A Batch object containing sampled transitions as JAX arrays.
        """
        if key is not None:
            indices = jax.random.randint(
                key, shape=(batch_size,), minval=0, maxval=self.size
            )
            indices = np.asarray(indices)
        else:
            indices = np.random.randint(0, self.size, size=batch_size)

        return Batch(
            obs=jnp.asarray(self.obs_buf[indices]),
            actions=jnp.asarray(self.action_buf[indices]),
            rewards=jnp.asarray(self.reward_buf[indices]),
            next_obs=jnp.asarray(self.next_obs_buf[indices]),
            dones=jnp.asarray(self.done_buf[indices]),
        )

    def __len__(self) -> int:
        """Returns the current number of stored transitions.

        Returns:
            The integer count of elements in the buffer.
        """
        return self.size

    def is_ready(self, batch_size: int) -> bool:
        """Checks if the buffer has enough samples for a batch.

        Args:
            batch_size: Desired batch size.

        Returns:
            True if the buffer size is greater than or equal to batch_size.
        """
        return self.size >= batch_size

    def save(self) -> Dict[str, Any]:
        """Saves the buffer state to a dictionary.

        Returns:
            A dictionary containing all buffer data arrays and metadata.
        """
        return {
            "obs_buf": self.obs_buf[: self.size],
            "action_buf": self.action_buf[: self.size],
            "reward_buf": self.reward_buf[: self.size],
            "next_obs_buf": self.next_obs_buf[: self.size],
            "done_buf": self.done_buf[: self.size],
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "capacity": self.max_size,
            "size": self.size,
        }

    def load(self, data: Dict[str, Any]) -> None:
        """Loads buffer state from a dictionary.

        Args:
            data: Dictionary containing buffer data from a previous save() call.
        """
        size = data["size"]
        self.obs_buf[:size] = data["obs_buf"]
        self.action_buf[:size] = data["action_buf"]
        self.reward_buf[:size] = data["reward_buf"]
        self.next_obs_buf[:size] = data["next_obs_buf"]
        self.done_buf[:size] = data["done_buf"]
        self.size = size
        self.ptr = size % self.max_size

    @classmethod
    def create(cls, obs_dim: int, act_dim: int, capacity: int) -> ReplayBuffer:
        """Factory method to create a new buffer.

        Args:
            obs_dim: Dimension of observations.
            act_dim: Dimension of actions.
            capacity: Maximum buffer size.

        Returns:
            A new instance of ReplayBuffer.
        """
        return cls(obs_dim=obs_dim, act_dim=act_dim, max_size=capacity)
