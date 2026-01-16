"""
Saliency-based gradient pruning for Lottery Ticket Hypothesis.

This module implements gradient-based pruning methods including:
- Taylor expansion (gradient * weight) saliency
- Pure gradient magnitude pruning
- Accumulated gradient statistics during training
"""

import jax
import jax.numpy as jnp
from typing import Optional, Dict, Any

from src.utils.types import Params, Batch


def prune_kernels_by_gradient_saliency(
    params: Params,
    gradients: Params,
    target_sparsity: float = 0.8,
    method: str = "taylor"
) -> Params:
    """
    Prune kernels based on gradient saliency (gradient * weight importance).
    
    This implements the Taylor expansion approximation of weight importance:
    importance(w) = |w * ∂L/∂w|
    
    Args:
        params: Model parameters (after training)
        gradients: Gradients of the loss w.r.t. parameters
        target_sparsity: Fraction of weights to prune (0.8 = 80% pruned)
        method: Pruning criterion
            - "taylor": |w * grad| (first-order Taylor expansion)
            - "gradient": |grad| only (pure gradient magnitude)
            - "magnitude": |w| only (standard magnitude pruning)
    
    Returns:
        Binary mask (1 = keep, 0 = prune)
    """
    flat_params_with_path, tree_def = jax.tree_util.tree_flatten_with_path(params)
    flat_grads_with_path, _ = jax.tree_util.tree_flatten_with_path(gradients)
    
    # Collect kernel saliency scores
    saliency_values = []
    
    for (path, param), (_, grad) in zip(flat_params_with_path, flat_grads_with_path):
        # Check if this is a kernel (weight) parameter
        is_kernel = any(
            isinstance(node, jax.tree_util.DictKey) and node.key == "kernel"
            for node in path
        )
        
        if is_kernel:
            # Compute saliency score based on method
            if method == "taylor":
                # Taylor expansion: importance = |w * grad|
                saliency = jnp.abs(param * grad).flatten()
            elif method == "gradient":
                # Pure gradient magnitude
                saliency = jnp.abs(grad).flatten()
            elif method == "magnitude":
                # Standard magnitude pruning (for comparison)
                saliency = jnp.abs(param).flatten()
            else:
                raise ValueError(f"Unknown method: {method}")
            
            saliency_values.append(saliency)
    
    # Concatenate all kernel saliency scores
    all_saliency = jnp.concatenate(saliency_values)
    
    # Determine threshold for target sparsity
    k = int(len(all_saliency) * target_sparsity)
    threshold = jnp.sort(all_saliency)[k]
    
    print(f"  > Global {method.capitalize()} Saliency Threshold: {threshold:.6e}")
    
    # Create mask based on saliency using tree structure
    flat_masks = []
    for (path, param), (_, grad) in zip(flat_params_with_path, flat_grads_with_path):
        is_kernel = any(
            isinstance(node, jax.tree_util.DictKey) and node.key == "kernel"
            for node in path
        )
        
        if is_kernel:
            # Compute saliency and prune
            if method == "taylor":
                saliency = jnp.abs(param * grad)
            elif method == "gradient":
                saliency = jnp.abs(grad)
            elif method == "magnitude":
                saliency = jnp.abs(param)
            
            mask_leaf = (saliency > threshold).astype(jnp.float32)
        else:
            # Keep biases intact
            mask_leaf = jnp.ones_like(param)
        
        flat_masks.append(mask_leaf)
    
    # Reconstruct mask tree
    mask = jax.tree_util.tree_unflatten(tree_def, flat_masks)
    
    return mask


def compute_gradients_from_batch(
    agent,
    batch: Batch,
    normalize_obs: bool = True
) -> tuple[Params, Params]:
    """
    Compute gradients for actor and critic from a single batch.
    
    This is useful for computing saliency when gradients weren't saved during training.
    
    Args:
        agent: SACAgent with trained parameters
        batch: Batch of replay buffer data
        normalize_obs: Whether to normalize observations
    
    Returns:
        (actor_gradients, critic_gradients) tuple
    """
    from src.agents.sac import _sac_train_step
    from functools import partial
    
    # Normalize observations if needed
    if normalize_obs:
        obs = agent.normalizer.normalize(batch.obs)
        next_obs = agent.normalizer.normalize(batch.next_obs)
        batch = Batch(
            obs=obs,
            actions=batch.actions,
            rewards=batch.rewards,
            next_obs=next_obs,
            dones=batch.dones
        )
    
    # Create gradient computation functions
    def actor_loss_fn(actor_params):
        """Compute actor loss for gradient calculation."""
        from src.networks.actor import sample_action
        
        # Get mean and log_std from actor
        mean, log_std = agent.state.actor_apply_fn(actor_params, batch.obs)
        
        # Sample actions
        key = jax.random.PRNGKey(0)  # Fixed key for deterministic gradients
        action, log_prob = sample_action(mean, log_std, key)
        
        # Compute Q-values
        q1, q2 = agent.state.critic_apply_fn(
            agent.state.critic_params,
            batch.obs,
            action
        )
        q_min = jnp.minimum(q1, q2)
        
        # Actor loss: maximize Q - alpha * entropy
        alpha = jnp.exp(agent.state.log_alpha)
        loss = (alpha * log_prob - q_min).mean()
        
        return loss
    
    def critic_loss_fn(critic_params):
        """Compute critic loss for gradient calculation."""
        
        # Compute current Q-values
        q1, q2 = agent.state.critic_apply_fn(
            critic_params,
            batch.obs,
            batch.actions
        )
        
        # Compute target Q-values (using target params)
        key = jax.random.PRNGKey(0)
        from src.networks.actor import sample_action
        
        # Get next action from actor
        next_mean, next_log_std = agent.state.actor_apply_fn(
            agent.state.actor_params,
            batch.next_obs
        )
        next_action, next_log_prob = sample_action(next_mean, next_log_std, key)
        
        target_q1, target_q2 = agent.state.critic_apply_fn(
            agent.state.target_critic_params,
            batch.next_obs,
            next_action
        )
        target_q = jnp.minimum(target_q1, target_q2)
        alpha = jnp.exp(agent.state.log_alpha)
        target_q = target_q - alpha * next_log_prob
        
        # Bellman backup
        target_q = batch.rewards + agent.config.gamma * (1 - batch.dones) * target_q
        target_q = jax.lax.stop_gradient(target_q)
        
        # MSE loss
        loss = ((q1 - target_q) ** 2).mean() + ((q2 - target_q) ** 2).mean()
        
        return loss
    
    # Compute gradients
    actor_grads = jax.grad(actor_loss_fn)(agent.state.actor_params)
    critic_grads = jax.grad(critic_loss_fn)(agent.state.critic_params)
    
    return actor_grads, critic_grads


def accumulate_gradient_statistics(
    agent,
    replay_buffer,
    num_batches: int = 100,
    batch_size: int = 256,
    normalize_obs: bool = True
) -> tuple[Params, Params]:
    """
    Accumulate gradient statistics over multiple batches.
    
    This provides more stable saliency estimates by averaging gradients
    across multiple mini-batches.
    
    Args:
        agent: SACAgent with trained parameters
        replay_buffer: ReplayBuffer to sample from
        num_batches: Number of batches to average over
        batch_size: Size of each batch
        normalize_obs: Whether to normalize observations
    
    Returns:
        (accumulated_actor_grads, accumulated_critic_grads) - averaged over batches
    """
    print(f"  > Accumulating gradients over {num_batches} batches...")
    
    # Initialize accumulators
    actor_grad_sum = jax.tree.map(jnp.zeros_like, agent.state.actor_params)
    critic_grad_sum = jax.tree.map(jnp.zeros_like, agent.state.critic_params)
    
    for i in range(num_batches):
        # Sample batch
        batch = replay_buffer.sample(batch_size)
        
        # Compute gradients for this batch
        actor_grads, critic_grads = compute_gradients_from_batch(
            agent, batch, normalize_obs
        )
        
        # Accumulate (take absolute value first for saliency)
        actor_grad_sum = jax.tree.map(
            lambda acc, g: acc + jnp.abs(g),
            actor_grad_sum,
            actor_grads
        )
        critic_grad_sum = jax.tree.map(
            lambda acc, g: acc + jnp.abs(g),
            critic_grad_sum,
            critic_grads
        )
        
        if (i + 1) % 25 == 0:
            print(f"    Processed {i + 1}/{num_batches} batches")
    
    # Average
    actor_grad_avg = jax.tree.map(lambda g: g / num_batches, actor_grad_sum)
    critic_grad_avg = jax.tree.map(lambda g: g / num_batches, critic_grad_sum)
    
    return actor_grad_avg, critic_grad_avg
