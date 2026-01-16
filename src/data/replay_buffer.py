from __future__ import annotations

from typing import Optional

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
        obs_dim: Dimension of observations.
        act_dim: Dimension of actions.
        max_size: Maximum number of transitions to store.
        size: Current number of transitions stored.
        ptr: Current write position in the circular buffer.

    Example:
        >>> buffer = ReplayBuffer(obs_dim=10, act_dim=4, max_size=100000)
        >>> buffer.store(obs, action, reward, next_obs, done)
        >>> batch = buffer.sample(batch_size=256)
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        max_size: int = 1_000_000,
    ):
        """Initialize the replay buffer.

        Args:
            obs_dim: Dimension of observations.
            act_dim: Dimension of actions.
            max_size: Maximum number of transitions to store.
        """
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_size = max_size

        # Preallocate arrays
        self.obs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.action_buf = np.zeros((max_size, act_dim), dtype=np.float32)
        self.reward_buf = np.zeros((max_size, 1), dtype=np.float32)
        self.done_buf = np.zeros((max_size, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def store(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: float,
    ) -> None:
        """Store a single transition in the buffer.

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
        """Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample.
            key: Optional JAX random key. If None, uses NumPy random.

        Returns:
            Batch named tuple containing JAX arrays.
        """
        if key is not None:
            # Use JAX random for reproducibility
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
        """Return current number of stored transitions."""
        return self.size

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for a batch.

        Args:
            batch_size: Desired batch size.

        Returns:
            True if buffer has at least batch_size transitions.
        """
        return self.size >= batch_size
