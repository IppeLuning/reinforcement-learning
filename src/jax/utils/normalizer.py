"""Running mean/variance normalizer for observations.

DISCLAIMER: This code was written by Claude Opus 4.5 on 2026-01-12
and reviewed by Marinus van den Ende.

This module provides online normalization of observations using Welford's
algorithm for numerically stable running statistics.
"""

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
        count: Number of observations seen.
        eps: Small constant for numerical stability.
        
    Example:
        >>> normalizer = RunningNormalizer.create(obs_dim=10)
        >>> normalizer, obs_norm = normalizer.normalize_and_update(obs)
    """
    
    mean: jax.Array
    var: jax.Array
    count: float
    eps: float = struct.field(pytree_node=False, default=1e-8)
    
    @classmethod
    def create(cls, obs_dim: int, eps: float = 1e-8) -> "RunningNormalizer":
        """Create a new normalizer initialized to identity transform.
        
        Args:
            obs_dim: Dimension of observations.
            eps: Small constant for numerical stability.
            
        Returns:
            Initialized RunningNormalizer.
        """
        return cls(
            mean=jnp.zeros(obs_dim),
            var=jnp.ones(obs_dim),
            count=0.0,
            eps=eps,
        )
    
    def update(self, batch_obs: jax.Array) -> "RunningNormalizer":
        """Update running statistics with a batch of observations.
        
        Uses Welford's parallel algorithm for numerically stable updates.
        
        Args:
            batch_obs: Batch of observations, shape (batch_size, obs_dim).
            
        Returns:
            Updated normalizer with new statistics.
        """
        # Handle single observation
        if batch_obs.ndim == 1:
            batch_obs = batch_obs[None, :]
        
        batch_count = float(batch_obs.shape[0])
        batch_mean = jnp.mean(batch_obs, axis=0)
        batch_var = jnp.var(batch_obs, axis=0)
        
        # Welford's parallel algorithm
        total_count = self.count + batch_count
        delta = batch_mean - self.mean
        
        new_mean = self.mean + delta * batch_count / jnp.maximum(total_count, 1.0)
        
        # Combine variances
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / jnp.maximum(total_count, 1.0)
        new_var = m2 / jnp.maximum(total_count, 1.0) + self.eps
        
        return self.replace(
            mean=new_mean,
            var=new_var,
            count=total_count,
        )
    
    def normalize(self, obs: jax.Array) -> jax.Array:
        """Normalize observations using current running statistics.
        
        Args:
            obs: Observations to normalize, shape (..., obs_dim).
            
        Returns:
            Normalized observations with approximately zero mean and unit variance.
        """
        return (obs - self.mean) / jnp.sqrt(self.var)
    
    def normalize_and_update(
        self, 
        batch_obs: jax.Array,
    ) -> Tuple["RunningNormalizer", jax.Array]:
        """Update statistics and normalize in one call.
        
        Args:
            batch_obs: Batch of observations to process.
            
        Returns:
            Tuple of (updated normalizer, normalized observations).
        """
        updated = self.update(batch_obs)
        normalized = updated.normalize(batch_obs)
        return updated, normalized
