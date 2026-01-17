"""
Step 2: Create Lottery Ticket Masks.

This module defines the 'create_mask' function called by run_pipeline.py.
It loads the final dense weights, applies magnitude pruning, and saves the masks.
"""

import os
import pickle
from typing import Any, Dict

import jax
import jax.numpy as jnp
import yaml

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


def create_mask(
    cfg: Dict[str, Any],
    task_name: str,
    seed: int,
    ckpt_dir: str,
    mask_out_path: str,
    pruning_method: str = "magnitude",
) -> None:
    """
    Executes Step 2 of the LTH pipeline:
    1. Loads 'checkpoint_final.pkl' from the dense run.
    2. Prunes Actor and Critic parameters using selected method.
    3. Saves the resulting masks to 'mask_out_path'.

    Args:
        cfg: Configuration dictionary.
        task_name: Name of the task.
        seed: Random seed.
        ckpt_dir: Directory containing checkpoints.
        mask_out_path: Path to save the mask.
        pruning_method: "magnitude" or "gradient" pruning.
    """

    # 1. Safety Checks
    if os.path.exists(mask_out_path):
        print(f"  [Skip] Mask already exists: {mask_out_path}")
        return

    print(
        f"  > Creating {pruning_method.upper()} Mask for {task_name} (Seed {seed})..."
    )

    # 2. Re-Initialize Agent (To get the correct State structure for loading)
    # We need to build the model structure to load weights into it.
    hp = cfg["hyperparameters"]
    defaults = hp["defaults"]
    task_overrides = hp.get("tasks", {}).get(task_name, {})
    params = {**defaults, **task_overrides}

    # Setup dummy env just to get shapes
    # (We don't need to run the env, just get dims)
    set_seed(seed)
    env, obs_dim, act_dim, act_low, act_high = make_metaworld_env(
        task_name, hp["max_episode_steps"], params["scale_rewards"], seed
    )

    # Initialize empty agent
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

    # 3. Load the FINAL Dense Weights
    checkpointer = Checkpointer(ckpt_dir)
    restored_state = checkpointer.restore(agent.state, item="checkpoint_best.pkl")

    if restored_state is None:
        raise FileNotFoundError(f"Could not find any checkpoints in {ckpt_dir}")

    agent.state = restored_state

    # 4. Prune Parameters
    target_sparsity = cfg.get("pruning", {}).get("sparsity", 0.8)  # Default 80%

    if pruning_method == "magnitude":
        # MAGNITUDE-BASED PRUNING (Fast, standard LTH)
        print(f"    Pruning Actor to {target_sparsity:.0%} sparsity (magnitude)...")
        actor_mask = prune_kernels_by_magnitude(
            restored_state.actor_params, target_sparsity
        )

        print(f"    Pruning Critic to {target_sparsity:.0%} sparsity (magnitude)...")
        critic_mask = prune_kernels_by_magnitude(
            restored_state.critic_params, target_sparsity
        )

    elif pruning_method == "gradient":
        # GRADIENT-BASED PRUNING (Better performance, requires replay buffer)
        gradient_method = cfg.get("pruning", {}).get("gradient_method", "taylor")
        num_gradient_batches = cfg.get("pruning", {}).get("num_gradient_batches", 100)
        gradient_batch_size = cfg.get("pruning", {}).get("gradient_batch_size", 256)

        # Load replay buffer
        replay_path = os.path.join(ckpt_dir, "replay_buffer.pkl")
        if not os.path.exists(replay_path):
            raise FileNotFoundError(
                f"Replay buffer not found at {replay_path}. "
                "Gradient pruning requires saved replay buffer from dense training."
            )

        print(f"    Loading replay buffer from {replay_path}")
        with open(replay_path, "rb") as f:
            replay_data = pickle.load(f)

        replay_buffer = ReplayBuffer.create(
            obs_dim=obs_dim,
            act_dim=act_dim,
            capacity=replay_data["capacity"],
        )
        replay_buffer.load(replay_data)
        print(f"    Replay buffer size: {replay_buffer.size}")

        # Compute gradients
        print(
            f"    Computing gradients ({gradient_method}) over {num_gradient_batches} batches..."
        )
        actor_grads, critic_grads = accumulate_gradient_statistics(
            agent=agent,
            replay_buffer=replay_buffer,
            num_batches=num_gradient_batches,
            batch_size=gradient_batch_size,
            normalize_obs=True,
        )

        # Prune using gradient saliency
        print(
            f"    Pruning Actor to {target_sparsity:.0%} sparsity ({gradient_method} saliency)..."
        )
        actor_mask = prune_kernels_by_gradient_saliency(
            params=restored_state.actor_params,
            gradients=actor_grads,
            target_sparsity=target_sparsity,
            method=gradient_method,
        )

        print(
            f"    Pruning Critic to {target_sparsity:.0%} sparsity ({gradient_method} saliency)..."
        )
        critic_mask = prune_kernels_by_gradient_saliency(
            params=restored_state.critic_params,
            gradients=critic_grads,
            target_sparsity=target_sparsity,
            method=gradient_method,
        )
    else:
        raise ValueError(f"Unknown pruning_method: {pruning_method}")

    # 5. Verify Sparsity
    act_sp = compute_sparsity(actor_mask)
    crit_sp = compute_sparsity(critic_mask)
    print(f"    > Actual Actor Sparsity: {act_sp:.2%}")
    print(f"    > Actual Critic Sparsity: {crit_sp:.2%}")

    # 6. Save Masks
    mask_data = {
        "actor": actor_mask,
        "critic": critic_mask,
        "sparsity_target": target_sparsity,
        "pruning_method": pruning_method,
        "task": task_name,
        "seed": seed,
    }

    # Ensure directory exists
    os.makedirs(os.path.dirname(mask_out_path), exist_ok=True)

    with open(mask_out_path, "wb") as f:
        pickle.dump(mask_data, f)

    print(f"  > Mask saved to {mask_out_path}")

    # Cleanup
    env.close()
