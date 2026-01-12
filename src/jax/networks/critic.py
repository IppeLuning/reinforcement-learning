"""Q-Network (Critic) implementations for Soft Actor-Critic.

DISCLAIMER: This code was written by Claude Opus 4.5 on 2026-01-12
and reviewed by Marinus van den Ende.

This module provides the value function approximators used in SAC:
- QNetwork: Single Q-function that maps (state, action) -> Q-value
- TwinQNetwork: Two independent Q-networks for double Q-learning
"""

from __future__ import annotations

from typing import Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from src.jax.networks.mlp import MLP


class QNetwork(nn.Module):
    """Q-function network that estimates action-values Q(s, a).
    
    Takes concatenated observations and actions as input and outputs
    a scalar Q-value estimate.
    
    Attributes:
        hidden_dims: Sequence of hidden layer dimensions.
        
    Example:
        >>> q_net = QNetwork(hidden_dims=(256, 256))
        >>> params = q_net.init(
        ...     jax.random.PRNGKey(0), 
        ...     jnp.ones((1, 10)),  # obs
        ...     jnp.ones((1, 4))    # action
        ... )
        >>> q_value = q_net.apply(params, obs, action)
    """
    
    hidden_dims: Sequence[int] = (256, 256)
    
    @nn.compact
    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        """Compute Q-value for observation-action pair.
        
        Args:
            obs: Observations of shape (batch_size, obs_dim).
            action: Actions of shape (batch_size, act_dim).
            
        Returns:
            Q-values of shape (batch_size, 1).
        """
        # Concatenate observations and actions
        x = jnp.concatenate([obs, action], axis=-1)
        
        # Pass through MLP to get Q-value
        q_value = MLP(
            hidden_dims=self.hidden_dims,
            output_dim=1,
        )(x)
        
        return q_value


class TwinQNetwork(nn.Module):
    """Twin Q-networks for double Q-learning in SAC.
    
    Maintains two independent Q-networks to reduce overestimation bias.
    The minimum of the two Q-values is used for the target computation.
    
    Attributes:
        hidden_dims: Sequence of hidden layer dimensions for both networks.
        
    Example:
        >>> twin_q = TwinQNetwork(hidden_dims=(256, 256))
        >>> params = twin_q.init(
        ...     jax.random.PRNGKey(0),
        ...     jnp.ones((1, 10)),
        ...     jnp.ones((1, 4))
        ... )
        >>> q1, q2 = twin_q.apply(params, obs, action)
    """
    
    hidden_dims: Sequence[int] = (256, 256)
    
    @nn.compact
    def __call__(
        self, 
        obs: jax.Array, 
        action: jax.Array
    ) -> Tuple[jax.Array, jax.Array]:
        """Compute Q-values from both Q-networks.
        
        Args:
            obs: Observations of shape (batch_size, obs_dim).
            action: Actions of shape (batch_size, act_dim).
            
        Returns:
            Tuple of Q-values (q1, q2), each of shape (batch_size, 1).
        """
        # Concatenate inputs once
        x = jnp.concatenate([obs, action], axis=-1)
        
        # First Q-network
        q1 = MLP(
            hidden_dims=self.hidden_dims,
            output_dim=1,
            name="q1",
        )(x)
        
        # Second Q-network (independent parameters)
        q2 = MLP(
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
    """Compute soft target Q-value using double Q-learning.
    
    Uses the minimum of the two Q-values minus the entropy bonus.
    
    Args:
        q1: Q-values from first network.
        q2: Q-values from second network.
        log_prob: Log probability of the action.
        alpha: Temperature parameter for entropy regularization.
        
    Returns:
        Soft target Q-value.
    """
    min_q = jnp.minimum(q1, q2)
    return min_q - alpha * log_prob
