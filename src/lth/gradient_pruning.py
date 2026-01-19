"""
Saliency-based gradient pruning for Lottery Ticket Hypothesis.

This module implements gradient-based pruning methods including:
- Taylor expansion (gradient * weight) saliency
- Pure gradient magnitude pruning
- Accumulated gradient statistics during training
"""

from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp

from src.utils.types import Batch, Params


def prune_kernels_by_gradient_saliency(
    params: Params,
    gradients: Params,
    target_sparsity: float = 0.8,
    method: str = "taylor",
    prev_mask: Optional[Params] = None,  # <--- Added Argument
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
        prev_mask: (Optional) Mask from previous round. Weights that are 0 here
                   will be forced to 0 in the new mask.

    Returns:
        Binary mask (1 = keep, 0 = prune)
    """
    flat_params_with_path, tree_def = jax.tree_util.tree_flatten_with_path(params)
    flat_grads_with_path, _ = jax.tree_util.tree_flatten_with_path(gradients)

    # Flatten previous mask if it exists
    if prev_mask is not None:
        flat_prev_mask, _ = jax.tree_util.tree_flatten(prev_mask)
    else:
        flat_prev_mask = [None] * len(flat_params_with_path)

    # Collect kernel saliency scores
    saliency_values = []

    for (path, param), (_, grad), prev_m_leaf in zip(
        flat_params_with_path, flat_grads_with_path, flat_prev_mask
    ):
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
                # Standard magnitude pruning
                saliency = jnp.abs(param).flatten()
            else:
                raise ValueError(f"Unknown method: {method}")

            # [ITERATIVE PRUNING LOGIC]
            # If we have a previous mask, we must ensure that previously pruned weights
            # are NOT considered for the threshold calculation (or effectively stay at the bottom).
            # We set their saliency to -1.0 (assuming normal saliency is >= 0).
            if prev_m_leaf is not None:
                # Flatten mask to match saliency shape
                flat_pm = prev_m_leaf.flatten()
                # Where mask is 0, set saliency to -1.0
                saliency = jnp.where(flat_pm == 0, -1.0, saliency)

            saliency_values.append(saliency)

    # Concatenate all kernel saliency scores
    all_saliency = jnp.concatenate(saliency_values)

    # Determine threshold for target sparsity
    # We want to prune the bottom k% of ALL weights.
    # The weights that were already 0 (saliency -1.0) will be at the very bottom
    # of the sorted list, so they will be "re-pruned" automatically.
    k = int(len(all_saliency) * target_sparsity)

    if k > 0:
        threshold = jnp.sort(all_saliency)[k]
    else:
        threshold = -2.0  # Prune nothing

    print(f"  > Global {method.capitalize()} Saliency Threshold: {threshold:.6e}")

    # Create mask based on saliency using tree structure
    flat_masks = []

    # We re-iterate to apply the threshold
    for (path, param), (_, grad), prev_m_leaf in zip(
        flat_params_with_path, flat_grads_with_path, flat_prev_mask
    ):
        is_kernel = any(
            isinstance(node, jax.tree_util.DictKey) and node.key == "kernel"
            for node in path
        )

        if is_kernel:
            # Re-compute saliency to apply threshold
            if method == "taylor":
                saliency = jnp.abs(param * grad)
            elif method == "gradient":
                saliency = jnp.abs(grad)
            elif method == "magnitude":
                saliency = jnp.abs(param)

            # Apply Iterative Logic locally as well for consistency
            if prev_m_leaf is not None:
                saliency = jnp.where(prev_m_leaf == 0, -1.0, saliency)

            # Strict inequality (> threshold) ensures we kill the -1.0s
            mask_leaf = (saliency > threshold).astype(jnp.float32)
        else:
            # Keep biases intact
            mask_leaf = jnp.ones_like(param)

        flat_masks.append(mask_leaf)

    # Reconstruct mask tree
    mask = jax.tree_util.tree_unflatten(tree_def, flat_masks)

    return mask


def compute_gradients_from_batch(
    agent, batch: Batch, normalize_obs: bool = True
) -> tuple[Params, Params]:
    """
    Compute gradients for actor and critic from a single batch.
    Args:
        agent: SACAgent with trained parameters
        batch: Batch of replay buffer data
        normalize_obs: Whether to normalize observations
    Returns:
        (actor_gradients, critic_gradients) tuple
    """
    from src.networks.actor import sample_action

    # Normalize observations if needed
    if normalize_obs:
        obs = agent.normalizer.normalize(batch.obs)
        next_obs = agent.normalizer.normalize(batch.next_obs)
        batch = Batch(
            obs=obs,
            actions=batch.actions,
            rewards=batch.rewards,
            next_obs=next_obs,
            dones=batch.dones,
        )

    # Create gradient computation functions
    def actor_loss_fn(actor_params):
        """Compute actor loss for gradient calculation."""
        # Get mean and log_std from actor
        mean, log_std = agent.state.actor_apply_fn(actor_params, batch.obs)

        # Sample actions
        key = jax.random.PRNGKey(0)  # Fixed key for deterministic gradients
        action, log_prob = sample_action(mean, log_std, key)

        # Compute Q-values
        q1, q2 = agent.state.critic_apply_fn(
            agent.state.critic_params, batch.obs, action
        )
        q_min = jnp.minimum(q1, q2)

        # Actor loss: maximize Q - alpha * entropy
        alpha = jnp.exp(agent.state.log_alpha)
        loss = (alpha * log_prob - q_min).mean()

        return loss

    def critic_loss_fn(critic_params):
        """Compute critic loss for gradient calculation."""
        # Compute current Q-values
        q1, q2 = agent.state.critic_apply_fn(critic_params, batch.obs, batch.actions)

        # Compute target Q-values (using target params)
        key = jax.random.PRNGKey(0)

        # Get next action from actor
        next_mean, next_log_std = agent.state.actor_apply_fn(
            agent.state.actor_params, batch.next_obs
        )
        next_action, next_log_prob = sample_action(next_mean, next_log_std, key)

        target_q1, target_q2 = agent.state.critic_apply_fn(
            agent.state.target_critic_params, batch.next_obs, next_action
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
    normalize_obs: bool = True,
) -> tuple[Params, Params]:
    """
    Accumulate gradient statistics over multiple batches.
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
            lambda acc, g: acc + jnp.abs(g), actor_grad_sum, actor_grads
        )
        critic_grad_sum = jax.tree.map(
            lambda acc, g: acc + jnp.abs(g), critic_grad_sum, critic_grads
        )

        if (i + 1) % 25 == 0:
            print(f"    Processed {i + 1}/{num_batches} batches")

    # Average
    actor_grad_avg = jax.tree.map(lambda g: g / num_batches, actor_grad_sum)
    critic_grad_avg = jax.tree.map(lambda g: g / num_batches, critic_grad_sum)

    print(f"  > Gradient accumulation complete!")
    return actor_grad_avg, critic_grad_avg
