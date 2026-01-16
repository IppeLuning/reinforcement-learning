"""Gaussian Actor network for Soft Actor-Critic (SAC).

DISCLAIMER: This code was written by Claude Opus 4.5 on 2026-01-12
and reviewed by Marinus van den Ende.

This module implements a stochastic policy that outputs a Gaussian distribution
over continuous actions, with tanh squashing to bound the action space.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from src.networks.mlp import MLP


class GaussianActor(nn.Module):
    """Gaussian policy network with tanh squashing for continuous action spaces.

    The actor outputs the mean and log standard deviation of a Gaussian distribution.
    Actions are sampled using the reparameterization trick and squashed through tanh
    to bound them to [-1, 1], then rescaled to the actual action bounds.

    Attributes:
        act_dim: Dimension of the action space.
        hidden_dims: Sequence of hidden layer dimensions.
        log_std_bounds: Tuple of (min, max) bounds for log standard deviation.
            Prevents numerical instability from extreme variance values.

    Example:
        >>> actor = GaussianActor(act_dim=4, hidden_dims=(256, 256))
        >>> params = actor.init(jax.random.PRNGKey(0), jnp.ones((1, 10)))
        >>> mean, log_std = actor.apply(params, jnp.ones((32, 10)))
    """

    act_dim: int
    hidden_dims: Sequence[int] = (256, 256)
    log_std_bounds: Tuple[float, float] = (-20.0, 2.0)

    @nn.compact
    def __call__(self, obs: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Compute mean and log standard deviation of the policy distribution.

        Args:
            obs: Observations of shape (batch_size, obs_dim).

        Returns:
            Tuple of:
                - mean: Mean of the Gaussian, shape (batch_size, act_dim).
                - log_std: Log standard deviation, shape (batch_size, act_dim).
        """
        # Output both mean and log_std through the same backbone
        net_output = MLP(
            hidden_dims=self.hidden_dims,
            output_dim=2 * self.act_dim,  # mean + log_std
        )(obs)

        # Split into mean and log_std
        mean, log_std = jnp.split(net_output, 2, axis=-1)

        # Bound log_std using tanh squashing to prevent extreme values
        log_std_min, log_std_max = self.log_std_bounds
        log_std = jnp.tanh(log_std)
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1.0)

        return mean, log_std


def sample_action(
    mean: jax.Array,
    log_std: jax.Array,
    keys: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """Sample actions using reparameterization (batched, JAX-safe).

    Args:
        mean: (B, act_dim)
        log_std: (B, act_dim)
        keys: (B, 2) PRNG keys (one per environment)

    Returns:
        action: (B, act_dim) in [-1, 1]
        log_prob: (B, 1)
    """
    std = jnp.exp(log_std)

    # Sample noise independently per environment
    noise = jax.vmap(
        lambda k: jax.random.normal(k, shape=mean.shape[1:])
    )(keys)

    z = mean + std * noise

    # Tanh squashing
    action = jnp.tanh(z)

    # Gaussian log-prob
    log_prob = -0.5 * (
        jnp.log(2 * jnp.pi)
        + 2 * log_std
        + ((z - mean) / std) ** 2
    )
    log_prob = jnp.sum(log_prob, axis=-1, keepdims=True)

    # Tanh correction
    log_prob -= jnp.sum(
        jnp.log(1.0 - action**2 + 1e-6),
        axis=-1,
        keepdims=True,
    )

    return action, log_prob


def deterministic_action(mean: jax.Array) -> jax.Array:
    """Compute deterministic action (for evaluation).

    Simply applies tanh to the mean without sampling.

    Args:
        mean: Mean of the Gaussian distribution.

    Returns:
        Deterministic action in [-1, 1].
    """
    return jnp.tanh(mean)


def scale_action(
    action: jax.Array,
    act_low: jax.Array,
    act_high: jax.Array,
) -> jax.Array:
    """Scale action from [-1, 1] to [act_low, act_high].

    Args:
        action: Action in [-1, 1].
        act_low: Lower bounds of action space.
        act_high: Upper bounds of action space.

    Returns:
        Scaled action in [act_low, act_high].
    """
    return act_low + (action + 1.0) * 0.5 * (act_high - act_low)
