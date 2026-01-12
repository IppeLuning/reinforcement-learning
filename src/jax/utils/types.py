"""Type aliases and dataclasses for JAX RL implementation.

DISCLAIMER: This code was written by Claude Opus 4.5 on 2026-01-12
and reviewed by Marinus van den Ende.

This module provides consistent type definitions used throughout the codebase,
improving readability and enabling static type checking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, NamedTuple

import jax
import jax.numpy as jnp
from flax.core import FrozenDict

# =============================================================================
# Type Aliases
# =============================================================================

# JAX random key type
PRNGKey = jax.Array

# Flax parameter pytree (nested dict of arrays)
Params = FrozenDict[str, Any]

# Generic pytree for masks (same structure as Params but with binary values)
Mask = FrozenDict[str, Any]

# Dictionary of scalar metrics for logging
Metrics = Dict[str, float]


# =============================================================================
# Data Structures
# =============================================================================


class Batch(NamedTuple):
    """A batch of transitions sampled from the replay buffer.
    
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
        low: Lower bounds for each action dimension.
        high: Upper bounds for each action dimension.
    """
    low: jax.Array
    high: jax.Array
    
    @classmethod
    def from_numpy(cls, low, high) -> "ActionBounds":
        """Create ActionBounds from numpy arrays."""
        return cls(
            low=jnp.asarray(low, dtype=jnp.float32),
            high=jnp.asarray(high, dtype=jnp.float32),
        )


@dataclass
class TransitionData:
    """Raw transition data before batching.
    
    Used for storing individual transitions in the replay buffer.
    """
    obs: jax.Array
    action: jax.Array
    reward: float
    next_obs: jax.Array
    done: float
