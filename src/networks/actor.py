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
            A tuple containing:
                - mean: Mean of the Gaussian, shape (batch_size, act_dim).
                - log_std: Log standard deviation, shape (batch_size, act_dim).
        """
        net_output = MLP(
            hidden_dims=self.hidden_dims,
            output_dim=2 * self.act_dim,
        )(obs)

        mean: jax.Array
        log_std: jax.Array
        mean, log_std = jnp.split(net_output, 2, axis=-1)

        log_std_min, log_std_max = self.log_std_bounds
        log_std = jnp.tanh(log_std)
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1.0)

        return mean, log_std


def sample_action(
    mean: jax.Array,
    log_std: jax.Array,
    key: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """Sample an action from the Gaussian policy using reparameterization.

    Uses the reparameterization trick: $$action = \tanh(mean + std \cdot noise)$$.
    This ensures the sampling process is differentiable. It also computes the
    log probability with the change-of-variables correction for the tanh squash.

    Args:
        mean: Mean of the Gaussian distribution.
        log_std: Log standard deviation of the distribution.
        key: JAX random key for sampling.

    Returns:
        A tuple containing:
            - action: Sampled action in [-1, 1], shape (batch_size, act_dim).
            - log_prob: Log probability of the action, shape (batch_size, 1).
    """
    std = jnp.exp(log_std)

    noise = jax.random.normal(key, shape=mean.shape)
    z = mean + std * noise

    action = jnp.tanh(z)

    log_prob = -0.5 * (jnp.log(2 * jnp.pi) + 2 * log_std + ((z - mean) / std) ** 2)
    log_prob = jnp.sum(log_prob, axis=-1, keepdims=True)

    log_prob = log_prob - jnp.sum(
        jnp.log(1.0 - action**2 + 1e-6), axis=-1, keepdims=True
    )

    return action, log_prob


def deterministic_action(mean: jax.Array) -> jax.Array:
    """Compute deterministic action for evaluation.

    Args:
        mean: Mean of the Gaussian distribution.

    Returns:
        The mean squashed into the [-1, 1] range via tanh.
    """
    return jnp.tanh(mean)


def scale_action(
    action: jax.Array,
    act_low: jax.Array,
    act_high: jax.Array,
) -> jax.Array:
    """Scales an action from the normalized [-1, 1] range to the environment bounds.

    Args:
        action: Action in the range [-1, 1].
        act_low: Lower bounds of the environment action space.
        act_high: Upper bounds of the environment action space.

    Returns:
        The action rescaled to [act_low, act_high].
    """
    return act_low + (action + 1.0) * 0.5 * (act_high - act_low)
