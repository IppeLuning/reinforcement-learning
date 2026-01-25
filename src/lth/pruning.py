from __future__ import annotations

from typing import List, Optional

import jax
import jax.numpy as jnp

from src.utils.types import Mask, Params


def prune_kernels_by_magnitude(
    params: Params,
    target_sparsity: float = 0.8,
    prev_mask: Optional[Mask] = None,
) -> Mask:
    """Performs global magnitude pruning on model kernels.

    This function implements research-grade Lottery Ticket Hypothesis (LTH) pruning.
    It identifies a global threshold across all 'kernel' parameters in the PyTree
    while leaving biases and non-kernel parameters untouched (fully kept).
    It supports iterative pruning by ensuring that weights already pruned in
    `prev_mask` remain pruned.

    Args:
        params: The model parameters (PyTree) to prune.
        target_sparsity: The fraction of weights to prune (e.g., 0.8 means 80% pruned).
        prev_mask: Optional mask from a previous pruning iteration.
            If provided, the new mask will be the intersection of the new
            magnitude mask and this previous mask.

    Returns:
        A PyTree matching the structure of `params` containing binary values
        (1.0 for kept, 0.0 for pruned).
    """
    flat_params_with_path, tree_def = jax.tree_util.tree_flatten_with_path(params)

    if prev_mask is not None:
        flat_prev_mask, _ = jax.tree_util.tree_flatten(prev_mask)
    else:
        flat_prev_mask = [None] * len(flat_params_with_path)

    kernel_values: List[jax.Array] = []
    for (path, param), prev_m_leaf in zip(flat_params_with_path, flat_prev_mask):
        is_kernel = any(
            isinstance(node, jax.tree_util.DictKey) and node.key == "kernel"
            for node in path
        )
        if is_kernel:
            if prev_m_leaf is not None:
                val = jnp.abs(param * prev_m_leaf).flatten()
            else:
                val = jnp.abs(param).flatten()

            kernel_values.append(val)

    all_kernels = jnp.concatenate(kernel_values)

    k = int(len(all_kernels) * target_sparsity)
    threshold = jnp.sort(all_kernels)[k]

    print(f"  > Global Kernel Threshold: {threshold:.6e}")

    flat_new_masks: List[jax.Array] = []

    for (path, param), prev_m_leaf in zip(flat_params_with_path, flat_prev_mask):
        is_kernel = any(
            isinstance(node, jax.tree_util.DictKey) and node.key == "kernel"
            for node in path
        )

        if is_kernel:
            new_leaf = (jnp.abs(param) > threshold).astype(jnp.float32)

            if prev_m_leaf is not None:
                new_leaf = new_leaf * prev_m_leaf

            flat_new_masks.append(new_leaf)
        else:
            flat_new_masks.append(jnp.ones_like(param))

    mask: Mask = jax.tree_util.tree_unflatten(tree_def, flat_new_masks)

    return mask


def compute_sparsity(mask: Mask) -> float:
    """Computes the global sparsity of a pruning mask.

    Sparsity is defined as the percentage of zeros in the mask relative
    to the total number of parameters.

    Args:
        mask: The binary pruning mask (PyTree).

    Returns:
        The sparsity as a float between 0.0 and 1.0.
    """
    flat_mask, _ = jax.tree.flatten(mask)
    total_params = sum(m.size for m in flat_mask)
    nonzero_params = sum(jnp.sum(m).item() for m in flat_mask)

    return 1.0 - (float(nonzero_params) / total_params)
