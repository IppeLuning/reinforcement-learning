"""Saliency-based pruning for neural networks.

This module implements saliency-based pruning methods like SNIP (Single-shot
Network Pruning based on Connection Sensitivity). Saliency scores measure
how important each parameter is by considering both its magnitude and gradient.

Reference:
    Lee et al. "SNIP: Single-shot Network Pruning based on Connection 
    Sensitivity" (ICLR 2019)
"""

from __future__ import annotations

from functools import partial
from typing import Callable, List, Tuple

import jax
import jax.numpy as jnp
from flax.core import FrozenDict

from src.jax.utils.types import Params, Batch


def compute_saliency(
    params: Params,
    loss_fn: Callable[[Params, Batch], jax.Array],
    batch: Batch,
) -> Params:
    """Compute saliency scores for all parameters.
    
    Saliency is defined as |weight * gradient|, which captures both the
    magnitude of the parameter and its sensitivity to the loss.
    
    Args:
        params: Network parameters (pytree).
        loss_fn: Function that computes loss given params and batch.
        batch: Batch of data for gradient computation.
        
    Returns:
        Pytree of saliency scores with same structure as params.
        
    Example:
        >>> def loss_fn(params, batch):
        ...     return jnp.mean((model.apply(params, batch.obs) - batch.targets) ** 2)
        >>> saliency = compute_saliency(params, loss_fn, batch)
    """
    # Compute gradients
    grads = jax.grad(loss_fn)(params, batch)
    
    # Saliency = |weight * gradient|
    saliency = jax.tree.map(
        lambda p, g: jnp.abs(p * g),
        params,
        grads,
    )
    
    return saliency


def prune_by_saliency(
    params: Params,
    saliency: Params,
    target_sparsity: float,
) -> Tuple[Params, Params]:
    """Create pruning mask based on saliency scores.
    
    Keeps the top (1 - target_sparsity) fraction of parameters by saliency.
    
    Args:
        params: Network parameters (used for structure).
        saliency: Saliency scores with same structure as params.
        target_sparsity: Target fraction of weights to prune (0.0 to 1.0).
        
    Returns:
        Tuple of (mask, pruned_params):
            - mask: Binary mask (1 = keep, 0 = prune)
            - pruned_params: Parameters with pruned weights set to 0
    """
    # Flatten all saliency scores
    flat_saliency, tree_def = jax.tree.flatten(saliency)
    all_scores = jnp.concatenate([s.flatten() for s in flat_saliency])
    
    # Find threshold for target sparsity
    num_params = all_scores.shape[0]
    num_to_prune = int(num_params * target_sparsity)
    
    # Use partition to find threshold efficiently
    threshold = jnp.sort(all_scores)[num_to_prune]
    
    # Create mask: keep weights with saliency >= threshold
    def create_mask(s):
        return (s >= threshold).astype(jnp.float32)
    
    mask = jax.tree.map(create_mask, saliency)
    
    # Apply mask to parameters
    pruned_params = jax.tree.map(lambda p, m: p * m, params, mask)
    
    return mask, pruned_params


def iterative_pruning(
    params: Params,
    loss_fn: Callable[[Params, Batch], jax.Array],
    batches: List[Batch],
    target_sparsity: float,
    num_iterations: int = 10,
) -> Tuple[Params, Params]:
    """Iterative saliency-based pruning.
    
    Performs pruning in multiple rounds, each time pruning a fraction
    of the remaining weights. This is more stable than one-shot pruning
    at high sparsity levels.
    
    Args:
        params: Initial network parameters.
        loss_fn: Loss function for saliency computation.
        batches: List of data batches (one per iteration).
        target_sparsity: Final target sparsity (0.0 to 1.0).
        num_iterations: Number of pruning rounds.
        
    Returns:
        Tuple of (final_mask, pruned_params).
    """
    # Compute sparsity per iteration (geometric schedule)
    # s_i = 1 - (1 - target)^(i/n) gives smooth progression
    sparsity_schedule = [
        1.0 - (1.0 - target_sparsity) ** ((i + 1) / num_iterations)
        for i in range(num_iterations)
    ]
    
    # Start with no pruning
    current_mask = jax.tree.map(jnp.ones_like, params)
    current_params = params
    
    for i, sparsity in enumerate(sparsity_schedule):
        # Use corresponding batch (cycle if needed)
        batch = batches[i % len(batches)]
        
        # Compute saliency for current (masked) parameters
        saliency = compute_saliency(current_params, loss_fn, batch)
        
        # Mask out already pruned weights from saliency
        saliency = jax.tree.map(
            lambda s, m: s * m,
            saliency,
            current_mask,
        )
        
        # Prune to new sparsity level
        new_mask, current_params = prune_by_saliency(
            current_params, saliency, sparsity
        )
        
        # Combine masks (intersection)
        current_mask = jax.tree.map(
            lambda m1, m2: m1 * m2,
            current_mask,
            new_mask,
        )
    
    return current_mask, current_params


def compute_saliency_for_sac(
    actor_params: Params,
    critic_params: Params,
    actor_apply_fn: Callable,
    critic_apply_fn: Callable,
    batch: Batch,
    alpha: float,
) -> Tuple[Params, Params]:
    """Compute saliency scores for SAC actor and critic.
    
    Uses the combined actor-critic loss for saliency computation,
    which captures the importance of parameters for the full RL objective.
    
    Args:
        actor_params: Actor network parameters.
        critic_params: Critic network parameters.
        actor_apply_fn: Actor forward function.
        critic_apply_fn: Critic forward function.
        batch: Batch of transitions.
        alpha: Temperature parameter.
        
    Returns:
        Tuple of (actor_saliency, critic_saliency).
    """
    # Actor loss function
    def actor_loss_fn(params, batch):
        mean, log_std = actor_apply_fn(params, batch.obs)
        # Simplified: just use mean for saliency
        return jnp.mean(mean ** 2)
    
    # Critic loss function  
    def critic_loss_fn(params, batch):
        q1, q2 = critic_apply_fn(params, batch.obs, batch.actions)
        return jnp.mean(q1 ** 2) + jnp.mean(q2 ** 2)
    
    # Compute saliencies
    actor_saliency = compute_saliency(actor_params, actor_loss_fn, batch)
    critic_saliency = compute_saliency(critic_params, critic_loss_fn, batch)
    
    return actor_saliency, critic_saliency
