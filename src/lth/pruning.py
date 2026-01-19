from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp


def prune_kernels_by_magnitude(
    params: Any,
    target_sparsity: float = 0.8,
    prev_mask: Optional[Any] = None,  # <--- Added Argument
) -> Any:
    """
    Research-Grade LTH Pruning: Prunes GLOBAL kernels only. Biases are kept.
    Supports Iterative Pruning via prev_mask.
    """
    # 1. Identify which leaves are kernels (weights) vs biases
    flat_params_with_path, tree_def = jax.tree_util.tree_flatten_with_path(params)

    # Flatten previous mask if it exists to align with params
    if prev_mask is not None:
        flat_prev_mask, _ = jax.tree_util.tree_flatten(prev_mask)
    else:
        # Create a dummy list of Nones if no mask exists
        flat_prev_mask = [None] * len(flat_params_with_path)

    # 2. Collect only KERNEL weights for threshold calculation
    kernel_values = []
    for (path, param), prev_m_leaf in zip(flat_params_with_path, flat_prev_mask):
        is_kernel = any(
            isinstance(node, jax.tree_util.DictKey) and node.key == "kernel"
            for node in path
        )
        if is_kernel:
            # If we have a previous mask, we can forcefully treat those weights as 0.0
            # (Though usually they are already 0.0 in params if trained with masking)
            if prev_m_leaf is not None:
                # Use the mask to ensure we are looking at the 'effective' weight
                val = jnp.abs(param * prev_m_leaf).flatten()
            else:
                val = jnp.abs(param).flatten()

            kernel_values.append(val)

    # Concatenate only kernels
    all_kernels = jnp.concatenate(kernel_values)

    # 3. Determine threshold from KERNELS only
    # Note: Weights that were previously pruned (0.0) will be at the bottom
    # of this sorted list, so they are automatically selected to be pruned again.
    k = int(len(all_kernels) * target_sparsity)
    threshold = jnp.sort(all_kernels)[k]

    print(f"  > Global Kernel Threshold: {threshold:.6e}")

    # 4. Create Mask (Apply pruning ONLY to kernels)
    # We cannot use tree_map_with_path easily with two trees (params + prev_mask),
    # so we iterate manually over the flattened lists and reconstruct.
    flat_new_masks = []

    for (path, param), prev_m_leaf in zip(flat_params_with_path, flat_prev_mask):
        is_kernel = any(
            isinstance(node, jax.tree_util.DictKey) and node.key == "kernel"
            for node in path
        )

        if is_kernel:
            # Prune if below threshold
            # Strict inequality (> threshold) ensures 0.0s remain 0
            new_leaf = (jnp.abs(param) > threshold).astype(jnp.float32)

            # Enforce previous mask (Iterative Safety Net)
            if prev_m_leaf is not None:
                new_leaf = new_leaf * prev_m_leaf

            flat_new_masks.append(new_leaf)
        else:
            # Keep biases/others 100% intact
            flat_new_masks.append(jnp.ones_like(param))

    # Reconstruct the tree
    mask = jax.tree_util.tree_unflatten(tree_def, flat_new_masks)

    return mask
