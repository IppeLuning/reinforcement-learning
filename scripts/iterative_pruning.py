"""
Iterative Pruning for Lottery Ticket Hypothesis.

Instead of pruning once to target sparsity (e.g., 80%), this module implements
gradual pruning over multiple iterations, which often discovers better subnetworks.

Example: To reach 80% sparsity in 4 iterations with prune_rate=0.2:
    Iteration 1: 20% → 80% remain
    Iteration 2: 20% of 80% → 64% remain
    Iteration 3: 20% of 64% → 51.2% remain
    Iteration 4: 20% of 51.2% → 41% remain... (adjust to hit target)

The key formula: remaining_weights = (1 - prune_rate) ^ num_iterations
"""

import os
import pickle
from typing import Any, Dict

import jax
import jax.numpy as jnp

from src.agents import SACAgent, SACConfig
from src.data import ReplayBuffer
from src.envs import make_metaworld_env
from src.lth import (
    accumulate_gradient_statistics,
    compute_sparsity,
    prune_kernels_by_gradient_saliency,
    prune_kernels_by_magnitude,
)
from src.utils import Checkpointer, set_seed


def compute_per_iteration_prune_rate(target_sparsity: float, num_iterations: int) -> float:
    """
    Calculate the per-iteration pruning rate needed to reach target sparsity.
    
    Formula: prune_rate = 1 - (1 - target_sparsity)^(1/n)
    
    Example:
        - Target: 80% sparsity (20% remain)
        - Iterations: 4
        - Result: prune_rate ≈ 0.331 (prune 33.1% each iteration)
    
    Args:
        target_sparsity: Final desired sparsity (e.g., 0.8 for 80%)
        num_iterations: Number of pruning iterations
        
    Returns:
        Per-iteration prune rate
    """
    remaining_fraction = 1.0 - target_sparsity
    per_iter_keep_rate = remaining_fraction ** (1.0 / num_iterations)
    per_iter_prune_rate = 1.0 - per_iter_keep_rate
    return per_iter_prune_rate


def prune_with_existing_mask(
    params,
    current_mask,
    target_sparsity_for_iteration: float,
    gradients=None,
    pruning_method: str = "magnitude",
    gradient_method: str = "taylor",
):
    """
    Prune parameters while respecting existing mask from previous iterations.
    
    Only prunes from weights that are currently active (mask == 1).
    
    Args:
        params: Current parameters
        current_mask: Binary mask from previous iteration (1 = active, 0 = pruned)
        target_sparsity_for_iteration: Sparsity to apply to active weights
        gradients: Gradient statistics (required for gradient-based pruning)
        pruning_method: "magnitude" or "gradient"
        gradient_method: "taylor", "gradient", or "magnitude" (for gradient pruning)
        
    Returns:
        Updated mask with additional pruning applied
    """
    flat_params_with_path, _ = jax.tree_util.tree_flatten_with_path(params)
    flat_mask_with_path, tree_def = jax.tree_util.tree_flatten_with_path(current_mask)
    
    # Collect saliency scores only from active weights (where mask == 1)
    saliency_values = []
    
    if pruning_method == "gradient":
        flat_grads_with_path, _ = jax.tree_util.tree_flatten_with_path(gradients)
    
    for idx, ((path, param), (_, mask_val)) in enumerate(zip(flat_params_with_path, flat_mask_with_path)):
        # Check if this is a kernel parameter
        is_kernel = any(
            isinstance(node, jax.tree_util.DictKey) and node.key == "kernel"
            for node in path
        )
        
        if is_kernel:
            # Get only active weights (where mask == 1)
            active_mask = mask_val == 1.0
            
            if pruning_method == "magnitude":
                # Standard magnitude pruning on active weights
                saliency = jnp.where(active_mask, jnp.abs(param), jnp.inf)
            elif pruning_method == "gradient":
                grad = flat_grads_with_path[idx][1]
                
                if gradient_method == "taylor":
                    saliency = jnp.where(active_mask, jnp.abs(param * grad), jnp.inf)
                elif gradient_method == "gradient":
                    saliency = jnp.where(active_mask, jnp.abs(grad), jnp.inf)
                elif gradient_method == "magnitude":
                    saliency = jnp.where(active_mask, jnp.abs(param), jnp.inf)
            
            saliency_values.append(saliency.flatten())
    
    # Concatenate saliency scores
    all_saliency = jnp.concatenate(saliency_values)
    
    # Count active weights (excluding inf placeholders)
    num_active = jnp.sum(jnp.isfinite(all_saliency))
    
    # Determine threshold to prune target_sparsity of active weights
    k = int(num_active * target_sparsity_for_iteration)
    
    # Filter out inf values and sort
    finite_saliency = all_saliency[jnp.isfinite(all_saliency)]
    threshold = jnp.sort(finite_saliency)[k] if k < len(finite_saliency) else 0.0
    
    print(f"    > Active weights: {num_active:.0f}, Pruning {k:.0f} ({target_sparsity_for_iteration:.1%})")
    print(f"    > Threshold: {threshold:.6e}")
    
    # Create updated mask
    flat_new_mask = []
    for idx, ((path, param), (_, mask_val)) in enumerate(zip(flat_params_with_path, flat_mask_with_path)):
        is_kernel = any(
            isinstance(node, jax.tree_util.DictKey) and node.key == "kernel"
            for node in path
        )
        
        if is_kernel:
            active_mask = mask_val == 1.0
            
            if pruning_method == "magnitude":
                saliency = jnp.abs(param)
            elif pruning_method == "gradient":
                grad = flat_grads_with_path[idx][1]
                if gradient_method == "taylor":
                    saliency = jnp.abs(param * grad)
                elif gradient_method == "gradient":
                    saliency = jnp.abs(grad)
                elif gradient_method == "magnitude":
                    saliency = jnp.abs(param)
            
            # Keep weight if: (1) currently active AND (2) above threshold
            new_mask = jnp.where(active_mask & (saliency > threshold), 1.0, 0.0)
            flat_new_mask.append(new_mask)
        else:
            # Keep biases intact
            flat_new_mask.append(mask_val)
    
    # Reconstruct mask tree
    new_mask = jax.tree_util.tree_unflatten(tree_def, flat_new_mask)
    
    return new_mask


def iterative_pruning(
    cfg: Dict[str, Any],
    task_name: str,
    seed: int,
    ckpt_dir: str,
    mask_out_path: str,
    pruning_method: str = "magnitude",
) -> None:
    """
    Perform iterative pruning to discover lottery tickets.
    
    Process:
    1. Load dense trained network
    2. For each iteration:
        a. Prune a fraction of remaining weights
        b. Evaluate intermediate ticket
        c. Save iteration checkpoint
    3. Save final mask
    
    Args:
        cfg: Configuration dictionary
        task_name: Environment task name
        seed: Random seed
        ckpt_dir: Checkpoint directory containing dense trained model
        mask_out_path: Where to save final mask
        pruning_method: "magnitude" or "gradient"
    """
    
    # Check if final mask already exists
    if os.path.exists(mask_out_path):
        print(f"  [Skip] Final mask already exists: {mask_out_path}")
        return
    
    print(f"\n{'='*70}")
    print(f"  ITERATIVE PRUNING: {task_name} (Seed {seed})")
    print(f"  Method: {pruning_method.upper()}")
    print(f"{'='*70}\n")
    
    # Load config
    hp = cfg["hyperparameters"]
    defaults = hp["defaults"]
    task_overrides = hp.get("tasks", {}).get(task_name, {})
    params = {**defaults, **task_overrides}
    
    pruning_cfg = cfg.get("pruning", {})
    target_sparsity = pruning_cfg.get("sparsity", 0.8)
    
    iterative_cfg = pruning_cfg.get("iterative", {})
    num_iterations = iterative_cfg.get("num_iterations", 4)
    
    # Calculate per-iteration prune rate
    per_iter_prune_rate = compute_per_iteration_prune_rate(target_sparsity, num_iterations)
    
    print(f"Target Sparsity: {target_sparsity:.1%}")
    print(f"Iterations: {num_iterations}")
    print(f"Per-Iteration Prune Rate: {per_iter_prune_rate:.2%}\n")
    
    # Setup environment and agent
    set_seed(seed)
    env, obs_dim, act_dim, act_low, act_high = make_metaworld_env(
        task_name, hp["max_episode_steps"], params["scale_rewards"], seed
    )
    
    sac_config = SACConfig(
        gamma=params.get("gamma", 0.99),
        tau=params.get("tau", 0.005),
        actor_lr=params.get("actor_lr", 3e-4),
        critic_lr=params.get("critic_lr", 3e-4),
        alpha_lr=params.get("alpha_lr", 3e-4),
        target_entropy_scale=params.get("target_entropy_scale", 1.0),
        auto_alpha=params.get("auto_alpha", True),
        init_alpha=params.get("init_alpha", 0.2),
        hidden_dims=tuple(hp["hidden_dims"]),
        use_masking=False,
    )
    
    agent = SACAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_low=act_low,
        act_high=act_high,
        config=sac_config,
        seed=seed,
    )
    
    # Load dense trained weights
    checkpointer = Checkpointer(ckpt_dir)
    restored_state = checkpointer.restore(agent.state, item="checkpoint_best.pkl")
    
    if restored_state is None:
        raise FileNotFoundError(f"Could not find checkpoint in {ckpt_dir}")
    
    agent.state = restored_state
    print(f"✓ Loaded dense weights from {ckpt_dir}\n")
    
    # Load replay buffer for gradient pruning
    replay_buffer = None
    if pruning_method == "gradient":
        replay_path = os.path.join(ckpt_dir, "replay_buffer.pkl")
        if not os.path.exists(replay_path):
            raise FileNotFoundError(
                f"Replay buffer required for gradient pruning: {replay_path}"
            )
        
        with open(replay_path, "rb") as f:
            replay_data = pickle.load(f)
        
        replay_buffer = ReplayBuffer.create(
            obs_dim=obs_dim,
            act_dim=act_dim,
            capacity=replay_data["capacity"],
        )
        replay_buffer.load(replay_data)
        print(f"✓ Loaded replay buffer (size: {replay_buffer.size})\n")
    
    # Initialize masks to all ones (no pruning yet)
    from src.lth.masks import create_ones_mask
    actor_mask = create_ones_mask(restored_state.actor_params)
    critic_mask = create_ones_mask(restored_state.critic_params)
    
    # Iterative pruning loop
    iteration_masks_dir = os.path.join(os.path.dirname(mask_out_path), "iterations")
    os.makedirs(iteration_masks_dir, exist_ok=True)
    
    for iteration in range(1, num_iterations + 1):
        print(f"\n{'─'*70}")
        print(f"  ITERATION {iteration}/{num_iterations}")
        print(f"{'─'*70}")
        
        # Compute gradients for this iteration (if using gradient pruning)
        actor_grads, critic_grads = None, None
        if pruning_method == "gradient":
            gradient_method = pruning_cfg.get("gradient_method", "taylor")
            num_gradient_batches = pruning_cfg.get("num_gradient_batches", 100)
            gradient_batch_size = pruning_cfg.get("gradient_batch_size", 256)
            
            print(f"  Computing {gradient_method} gradients...")
            actor_grads, critic_grads = accumulate_gradient_statistics(
                agent=agent,
                replay_buffer=replay_buffer,
                num_batches=num_gradient_batches,
                batch_size=gradient_batch_size,
                normalize_obs=True,
            )
        
        # Prune actor
        print(f"  Pruning Actor...")
        actor_mask = prune_with_existing_mask(
            params=restored_state.actor_params,
            current_mask=actor_mask,
            target_sparsity_for_iteration=per_iter_prune_rate,
            gradients=actor_grads,
            pruning_method=pruning_method,
            gradient_method=pruning_cfg.get("gradient_method", "taylor"),
        )
        
        # Prune critic
        print(f"  Pruning Critic...")
        critic_mask = prune_with_existing_mask(
            params=restored_state.critic_params,
            current_mask=critic_mask,
            target_sparsity_for_iteration=per_iter_prune_rate,
            gradients=critic_grads,
            pruning_method=pruning_method,
            gradient_method=pruning_cfg.get("gradient_method", "taylor"),
        )
        
        # Compute current sparsity
        actor_sp = compute_sparsity(actor_mask)
        critic_sp = compute_sparsity(critic_mask)
        
        print(f"\n  Cumulative Sparsity:")
        print(f"    Actor:  {actor_sp:.2%}")
        print(f"    Critic: {critic_sp:.2%}")
        
        # Save intermediate mask
        iter_mask_path = os.path.join(
            iteration_masks_dir, 
            f"mask_iter_{iteration}_{pruning_method}.pkl"
        )
        
        iteration_mask_data = {
            "actor": actor_mask,
            "critic": critic_mask,
            "iteration": iteration,
            "actor_sparsity": actor_sp,
            "critic_sparsity": critic_sp,
            "pruning_method": pruning_method,
            "task": task_name,
            "seed": seed,
        }
        
        with open(iter_mask_path, "wb") as f:
            pickle.dump(iteration_mask_data, f)
        
        print(f"  ✓ Saved iteration {iteration} mask")
    
    # Save final mask
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  Target Sparsity: {target_sparsity:.2%}")
    print(f"  Actual Actor Sparsity:  {actor_sp:.2%}")
    print(f"  Actual Critic Sparsity: {critic_sp:.2%}")
    
    final_mask_data = {
        "actor": actor_mask,
        "critic": critic_mask,
        "sparsity_target": target_sparsity,
        "actual_sparsity": {"actor": actor_sp, "critic": critic_sp},
        "pruning_method": pruning_method,
        "iterative": True,
        "num_iterations": num_iterations,
        "task": task_name,
        "seed": seed,
    }
    
    os.makedirs(os.path.dirname(mask_out_path), exist_ok=True)
    with open(mask_out_path, "wb") as f:
        pickle.dump(final_mask_data, f)
    
    print(f"\n✓ Final mask saved to {mask_out_path}\n")
    
    env.close()
