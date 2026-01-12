"""Multi-Layer Perceptron (MLP) module implemented in Flax.

This module provides a flexible MLP building block used by actor and critic networks.
"""

from __future__ import annotations

from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn


class MLP(nn.Module):
    """A Multi-Layer Perceptron with configurable hidden layers and activation.
    
    This is the fundamental building block for both actor and critic networks.
    Each hidden layer consists of a Dense layer followed by an activation function.
    The output layer is a Dense layer without activation.
    
    Attributes:
        hidden_dims: Sequence of hidden layer dimensions.
        output_dim: Dimension of the output layer.
        activation: Activation function applied after each hidden layer.
        kernel_init: Initializer for Dense layer weights.
        bias_init: Initializer for Dense layer biases.
        
    Example:
        >>> mlp = MLP(hidden_dims=(256, 256), output_dim=10)
        >>> params = mlp.init(jax.random.PRNGKey(0), jnp.ones((1, 5)))
        >>> output = mlp.apply(params, jnp.ones((32, 5)))
        >>> output.shape
        (32, 10)
    """
    
    hidden_dims: Sequence[int] = (256, 256)
    output_dim: int = 1
    activation: Callable[[jax.Array], jax.Array] = nn.relu
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros
    
    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass through the MLP.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim).
            
        Returns:
            Output tensor of shape (batch_size, output_dim).
        """
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(
                hidden_dim,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )(x)
            x = self.activation(x)
        
        # Output layer (no activation)
        x = nn.Dense(
            self.output_dim,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
        )(x)
        
        return x
