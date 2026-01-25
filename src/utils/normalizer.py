from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from flax import struct


class RunningNormalizer(struct.PyTreeNode):
    """Running mean/variance normalizer using Welford's online algorithm.

    Maintains running statistics of observations and normalizes new observations
    to have approximately zero mean and unit variance. This is important for
    stable RL training, especially with different observation scales.

    All operations are JAX-compatible and can be JIT-compiled.

    Attributes:
        mean: Running mean of observations, shape (obs_dim,).
        var: Running variance of observations, shape (obs_dim,).
        count: Total number of observation samples seen.
        eps: Small constant for numerical stability (prevents division by zero).
    """

    mean: jax.Array
    var: jax.Array
    count: float
    eps: float = struct.field(pytree_node=False, default=1e-8)

    @classmethod
    def create(cls, obs_dim: int, eps: float = 1e-8) -> RunningNormalizer:
        """Creates a new normalizer initialized to identity transform.

        Args:
            obs_dim: Dimension of observations.
            eps: Small constant for numerical stability.

        Returns:
            An initialized RunningNormalizer instance.
        """
        return cls(
            mean=jnp.zeros(obs_dim),
            var=jnp.ones(obs_dim),
            count=0.0,
            eps=eps,
        )

    def update(self, batch_obs: jax.Array) -> RunningNormalizer:
        """Updates running statistics with a batch of observations.

        Uses Welford's parallel algorithm for numerically stable updates by
        combining current statistics with the batch's statistics.

        Args:
            batch_obs: Batch of observations, shape (batch_size, obs_dim).

        Returns:
            An updated RunningNormalizer instance with refined statistics.
        """
        if batch_obs.ndim == 1:
            batch_obs = batch_obs[None, :]

        batch_count: float = float(batch_obs.shape[0])
        batch_mean: jax.Array = jnp.mean(batch_obs, axis=0)
        batch_var: jax.Array = jnp.var(batch_obs, axis=0)

        total_count: float = self.count + batch_count
        delta: jax.Array = batch_mean - self.mean

        new_mean: jax.Array = self.mean + delta * batch_count / jnp.maximum(
            total_count, 1.0
        )

        m_a: jax.Array = self.var * self.count
        m_b: jax.Array = batch_var * batch_count
        m2: jax.Array = (
            m_a
            + m_b
            + delta**2 * self.count * batch_count / jnp.maximum(total_count, 1.0)
        )
        new_var: jax.Array = m2 / jnp.maximum(total_count, 1.0) + self.eps

        return self.replace(
            mean=new_mean,
            var=new_var,
            count=total_count,
        )

    def normalize(self, obs: jax.Array) -> jax.Array:
        """Normalizes observations using current running statistics.

        Args:
            obs: Observations to normalize, shape (..., obs_dim).

        Returns:
            Normalized observations with approximately zero mean and unit variance.
        """
        return (obs - self.mean) / jnp.sqrt(self.var)

    def normalize_and_update(
        self,
        batch_obs: jax.Array,
    ) -> Tuple[RunningNormalizer, jax.Array]:
        """Updates statistics and normalizes the input batch in a single call.

        Args:
            batch_obs: Batch of observations to process.

        Returns:
            A tuple containing the updated RunningNormalizer and normalized observations.
        """
        updated = self.update(batch_obs)
        normalized = updated.normalize(batch_obs)
        return updated, normalized
