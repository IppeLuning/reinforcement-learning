import jax
import jax.numpy as jnp


def prune_kernels_by_magnitude(params, target_sparsity=0.8):
    """
    Research-Grade LTH Pruning: Prunes GLOBAL kernels only. Biases are kept.
    """
    # 1. Identify which leaves are kernels (weights) vs biases
    # We flatten with paths to filter "kernel"
    flat_params_with_path, tree_def = jax.tree_util.tree_flatten_with_path(params)

    # 2. Collect only KERNEL weights for threshold calculation
    kernel_values = []
    for path, param in flat_params_with_path:
        # Check if 'kernel' is in the path key (standard Flax naming)
        is_kernel = any(
            isinstance(node, jax.tree_util.DictKey) and node.key == "kernel"
            for node in path
        )
        if is_kernel:
            kernel_values.append(jnp.abs(param).flatten())

    # Concatenate only kernels
    all_kernels = jnp.concatenate(kernel_values)

    # 3. Determine threshold from KERNELS only
    k = int(len(all_kernels) * target_sparsity)
    threshold = jnp.sort(all_kernels)[k]

    print(f"  > Global Kernel Threshold: {threshold:.6f}")

    # 4. Create Mask (Apply pruning ONLY to kernels)
    def create_selective_mask(path, param):
        is_kernel = any(
            isinstance(node, jax.tree_util.DictKey) and node.key == "kernel"
            for node in path
        )

        if is_kernel:
            # Prune if below threshold
            return (jnp.abs(param) > threshold).astype(jnp.float32)
        else:
            # Keep biases/others 100% intact
            return jnp.ones_like(param)

    # Use map_with_path to apply logic
    mask = jax.tree_util.tree_map_with_path(create_selective_mask, params)

    return mask
