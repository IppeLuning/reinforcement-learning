from __future__ import annotations

from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from src.networks.mlp import MLP


class QNetwork(nn.Module):
    """Q-function network that estimates action-values $Q(s, a)$.

    Takes concatenated observations and actions as input and outputs
    a scalar Q-value estimate representing the expected return.

    Attributes:
        hidden_dims: Sequence of hidden layer dimensions for the MLP.
    """

    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        """Computes the Q-value for a given observation-action pair.

        Args:
            obs: Observations of shape (batch_size, obs_dim).
            action: Actions of shape (batch_size, act_dim).

        Returns:
            Q-values of shape (batch_size, 1).
        """
        x: jax.Array = jnp.concatenate([obs, action], axis=-1)

        q_value: jax.Array = MLP(
            hidden_dims=self.hidden_dims,
            output_dim=1,
        )(x)

        return q_value


class TwinQNetwork(nn.Module):
    """Twin Q-networks for double Q-learning in SAC.

    Maintains two independent Q-networks to reduce overestimation bias.
    The minimum of the two Q-values is used for the target computation
    to provide a more conservative estimate of the value function.

    Attributes:
        hidden_dims: Sequence of hidden layer dimensions for both networks.
    """

    hidden_dims: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(
        self, obs: jax.Array, action: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        """Computes Q-values from both independent Q-networks.

        Args:
            obs: Observations of shape (batch_size, obs_dim).
            action: Actions of shape (batch_size, act_dim).

        Returns:
            A tuple containing (q1, q2), each of shape (batch_size, 1).
        """
        x: jax.Array = jnp.concatenate([obs, action], axis=-1)

        q1: jax.Array = MLP(
            hidden_dims=self.hidden_dims,
            output_dim=1,
            name="q1",
        )(x)

        q2: jax.Array = MLP(
            hidden_dims=self.hidden_dims,
            output_dim=1,
            name="q2",
        )(x)

        return q1, q2


def compute_target_q(
    q1: jax.Array,
    q2: jax.Array,
    log_prob: jax.Array,
    alpha: jax.Array,
) -> jax.Array:
    """Computes the soft target Q-value using clipped double Q-learning.

    This function implements the SAC target calculation:
    $$Q_{target} = \min(Q_1, Q_2) - \alpha \cdot \log \pi(a|s)$$

    Args:
        q1: Q-values from the first critic network.
        q2: Q-values from the second critic network.
        log_prob: Log probability of the sampled action under the current policy.
        alpha: Temperature parameter controlling the entropy/exploration trade-off.

    Returns:
        The soft target Q-value used for the critic loss.
    """
    min_q: jax.Array = jnp.minimum(q1, q2)
    return min_q - alpha * log_prob
