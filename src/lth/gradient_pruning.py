from __future__ import annotations

from typing import Any, List, Optional, Tuple

import jax
import jax.numpy as jnp

from src.utils.types import Batch, Params


def prune_kernels_by_gradient_saliency(
    params: Params,
    gradients: Params,
    target_sparsity: float = 0.8,
    method: str = "taylor",
    prev_mask: Optional[Params] = None,
) -> Params:
    """Prunes kernels based on gradient saliency (gradient * weight importance).

    This implements the Taylor expansion approximation of weight importance:
    $$importance(w) = |w * \partial L/\partial w|$$

    Args:
        params: Model parameters (after training).
        gradients: Gradients of the loss w.r.t. parameters.
        target_sparsity: Fraction of weights to prune (e.g., 0.8 = 80% pruned).
        method: Pruning criterion.
            - "taylor": |w * grad| (first-order Taylor expansion).
            - "gradient": |grad| only (pure gradient magnitude).
            - "magnitude": |w| only (standard magnitude pruning).
        prev_mask: Mask from previous round. Weights that are 0 here
            will be forced to 0 in the new mask.

    Returns:
        A binary mask (1.0 = keep, 0.0 = prune) as a PyTree matching params structure.

    Raises:
        ValueError: If an unknown pruning method is specified.
    """
    flat_params_with_path, tree_def = jax.tree_util.tree_flatten_with_path(params)
    flat_grads_with_path, _ = jax.tree_util.tree_flatten_with_path(gradients)

    if prev_mask is not None:
        flat_prev_mask, _ = jax.tree_util.tree_flatten(prev_mask)
    else:
        flat_prev_mask = [None] * len(flat_params_with_path)

    saliency_values: List[jax.Array] = []

    for (path, param), (_, grad), prev_m_leaf in zip(
        flat_params_with_path, flat_grads_with_path, flat_prev_mask
    ):
        is_kernel = any(
            isinstance(node, jax.tree_util.DictKey) and node.key == "kernel"
            for node in path
        )

        if is_kernel:
            if method == "taylor":
                saliency = jnp.abs(param * grad).flatten()
            elif method == "gradient":
                saliency = jnp.abs(grad).flatten()
            elif method == "magnitude":
                saliency = jnp.abs(param).flatten()
            else:
                raise ValueError(f"Unknown method: {method}")

            if prev_m_leaf is not None:
                flat_pm = prev_m_leaf.flatten()
                saliency = jnp.where(flat_pm == 0, -1.0, saliency)

            saliency_values.append(saliency)

    all_saliency = jnp.concatenate(saliency_values)

    k = int(len(all_saliency) * target_sparsity)

    if k > 0:
        threshold = jnp.sort(all_saliency)[k]
    else:
        threshold = -2.0

    print(f"  > Global {method.capitalize()} Saliency Threshold: {threshold:.6e}")

    flat_masks: List[jax.Array] = []

    for (path, param), (_, grad), prev_m_leaf in zip(
        flat_params_with_path, flat_grads_with_path, flat_prev_mask
    ):
        is_kernel = any(
            isinstance(node, jax.tree_util.DictKey) and node.key == "kernel"
            for node in path
        )

        if is_kernel:
            if method == "taylor":
                saliency = jnp.abs(param * grad)
            elif method == "gradient":
                saliency = jnp.abs(grad)
            elif method == "magnitude":
                saliency = jnp.abs(param)

            if prev_m_leaf is not None:
                saliency = jnp.where(prev_m_leaf == 0, -1.0, saliency)

            mask_leaf = (saliency > threshold).astype(jnp.float32)
        else:
            mask_leaf = jnp.ones_like(param)

        flat_masks.append(mask_leaf)

    mask = jax.tree_util.tree_unflatten(tree_def, flat_masks)
    return mask


def compute_gradients_from_batch(
    agent: Any, batch: Batch, normalize_obs: bool = True
) -> Tuple[Params, Params]:
    """Computes gradients for actor and critic from a single batch.

    Args:
        agent: SACAgent instance containing current state and config.
        batch: A batch of transitions from the replay buffer.
        normalize_obs: Whether to apply running normalization to observations.

    Returns:
        A tuple of (actor_gradients, critic_gradients).
    """
    from src.networks.actor import sample_action

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

    def actor_loss_fn(actor_params: Params) -> jax.Array:
        """Internal function to compute actor loss for differentiation."""
        mean, log_std = agent.state.actor_apply_fn(actor_params, batch.obs)

        key = jax.random.PRNGKey(0)
        action, log_prob = sample_action(mean, log_std, key)

        q1, q2 = agent.state.critic_apply_fn(
            agent.state.critic_params, batch.obs, action
        )
        q_min = jnp.minimum(q1, q2)

        alpha = jnp.exp(agent.state.log_alpha)
        loss = (alpha * log_prob - q_min).mean()

        return loss

    def critic_loss_fn(critic_params: Params) -> jax.Array:
        """Internal function to compute critic loss for differentiation."""
        q1, q2 = agent.state.critic_apply_fn(critic_params, batch.obs, batch.actions)

        key = jax.random.PRNGKey(0)

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

        target_q = batch.rewards + agent.config.gamma * (1 - batch.dones) * target_q
        target_q = jax.lax.stop_gradient(target_q)

        loss = ((q1 - target_q) ** 2).mean() + ((q2 - target_q) ** 2).mean()

        return loss

    actor_grads = jax.grad(actor_loss_fn)(agent.state.actor_params)
    critic_grads = jax.grad(critic_loss_fn)(agent.state.critic_params)

    return actor_grads, critic_grads


def accumulate_gradient_statistics(
    agent: Any,
    replay_buffer: Any,
    num_batches: int = 100,
    batch_size: int = 256,
    normalize_obs: bool = True,
) -> Tuple[Params, Params]:
    """Accumulates absolute gradient statistics over multiple data batches.

    Args:
        agent: SACAgent instance.
        replay_buffer: Buffer to sample training data from.
        num_batches: Number of random batches to aggregate.
        batch_size: Size of each sampled batch.
        normalize_obs: Whether to normalize observations in the batches.

    Returns:
        A tuple of (avg_actor_gradients, avg_critic_gradients) where each
        leaf is the mean absolute gradient value.
    """
    print(f"  > Accumulating gradients over {num_batches} batches...")

    actor_grad_sum = jax.tree.map(jnp.zeros_like, agent.state.actor_params)
    critic_grad_sum = jax.tree.map(jnp.zeros_like, agent.state.critic_params)

    for i in range(num_batches):
        batch = replay_buffer.sample(batch_size)

        actor_grads, critic_grads = compute_gradients_from_batch(
            agent, batch, normalize_obs
        )

        actor_grad_sum = jax.tree.map(
            lambda acc, g: acc + jnp.abs(g), actor_grad_sum, actor_grads
        )
        critic_grad_sum = jax.tree.map(
            lambda acc, g: acc + jnp.abs(g), critic_grad_sum, critic_grads
        )

        if (i + 1) % 25 == 0:
            print(f"    Processed {i + 1}/{num_batches} batches")

    actor_grad_avg = jax.tree.map(lambda g: g / num_batches, actor_grad_sum)
    critic_grad_avg = jax.tree.map(lambda g: g / num_batches, critic_grad_sum)

    print(f"  > Gradient accumulation complete!")
    return actor_grad_avg, critic_grad_avg
