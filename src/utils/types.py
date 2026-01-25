from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, NamedTuple

import jax
import jax.numpy as jnp
from flax.core import FrozenDict

PRNGKey = jax.Array
"""A JAX random key array."""

Params = FrozenDict[str, Any]
"""A PyTree of model parameters (typically from Flax)."""

Mask = FrozenDict[str, Any]
"""A PyTree of binary values (0.0 or 1.0) used for weight pruning."""

Metrics = Dict[str, float]
"""A dictionary mapping metric names to their scalar values."""


class Batch(NamedTuple):
    """A batch of transitions sampled from the replay buffer.

    This structure is a NamedTuple to ensure it is treated as a valid JAX PyTree leaf
    or container, making it compatible with JIT-compiled functions.

    Attributes:
        obs: Observations of shape (batch_size, obs_dim).
        actions: Actions taken of shape (batch_size, act_dim).
        rewards: Scalar rewards of shape (batch_size, 1).
        next_obs: Next observations of shape (batch_size, obs_dim).
        dones: Terminal flags of shape (batch_size, 1).
    """

    obs: jax.Array
    actions: jax.Array
    rewards: jax.Array
    next_obs: jax.Array
    dones: jax.Array


@dataclass(frozen=True)
class ActionBounds:
    """Action space bounds for continuous control.

    Attributes:
        low: Lower bounds for each action dimension as a JAX array.
        high: Upper bounds for each action dimension as a JAX array.
    """

    low: jax.Array
    high: jax.Array

    @classmethod
    def from_numpy(cls, low: Any, high: Any) -> ActionBounds:
        """Creates ActionBounds from numpy or array-like inputs.

        Args:
            low: Array-like object representing lower bounds.
            high: Array-like object representing upper bounds.

        Returns:
            An instance of ActionBounds containing JAX float32 arrays.
        """
        return cls(
            low=jnp.asarray(low, dtype=jnp.float32),
            high=jnp.asarray(high, dtype=jnp.float32),
        )


@dataclass
class TransitionData:
    """Raw transition data before batching.

    Used for storing individual transitions in the replay buffer before they
    are stacked into a Batch.

    Attributes:
        obs: The environment observation array.
        action: The action taken by the agent.
        reward: The scalar reward received.
        next_obs: The subsequent environment observation.
        done: A float flag (1.0 for true, 0.0 for false) indicating episode termination.
    """

    obs: jax.Array
    action: jax.Array
    reward: float
    next_obs: jax.Array
    done: float
