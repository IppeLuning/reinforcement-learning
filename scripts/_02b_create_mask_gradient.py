"""
Script 2b: Create Pruning Masks with GRADIENT SALIENCY

This script creates pruning masks using gradient-based importance rather than
pure magnitude pruning. This implements the Taylor expansion saliency:
    importance(w) = |w * ∂L/∂w|

Usage:
    python scripts/_02b_create_mask_gradient.py

What this does:
1. Load trained dense models from checkpoints
2. Compute gradients using replay buffer data
3. Prune based on gradient saliency (weight * gradient magnitude)
4. Save masks for use in lottery ticket training
"""

import os
import pickle
import sys
from pathlib import Path

import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.sac import SACAgent, SACConfig
from src.data.replay_buffer import ReplayBuffer
from src.envs.factory import make_env
from src.lth import (
    compute_sparsity,
    prune_kernels_by_gradient_saliency,
    accumulate_gradient_statistics,
)
from src.utils.checkpointing import Checkpointer
from src.utils.jax_setup import configure_jax
from src.utils.seed import set_global_seed


def main():
    """Create gradient-based pruning masks."""
    configure_jax()

    # Load config
    config_path = project_root / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    tasks = cfg["tasks"]
    seeds = cfg["seeds"]
    base_dir = Path(cfg["base_dir"])

    # Gradient pruning settings
    pruning_config = cfg.get("pruning", {})
    target_sparsity = pruning_config.get("sparsity", 0.8)
    gradient_method = pruning_config.get("gradient_method", "taylor")  # "taylor", "gradient", or "magnitude"
    num_gradient_batches = pruning_config.get("num_gradient_batches", 100)
    gradient_batch_size = pruning_config.get("gradient_batch_size", 256)

    print("=" * 80)
    print("CREATING GRADIENT-BASED PRUNING MASKS")
    print("=" * 80)
    print(f"Method: {gradient_method}")
    print(f"Target Sparsity: {target_sparsity:.0%}")
    print(f"Gradient Batches: {num_gradient_batches}")
    print()

    for task_name in tasks:
        for seed in seeds:
            print(f"Processing: {task_name} | Seed {seed}")

            # Paths
            dense_dir = base_dir / task_name / f"seed_{seed}" / "dense"
            ckpt_dir = dense_dir / "checkpoints"
            replay_path = dense_dir / "replay_buffer.pkl"
            mask_out_path = (
                base_dir / task_name / f"seed_{seed}" / "mask_gradient.pkl"
            )

            if not ckpt_dir.exists():
                print(f"  [SKIP] No dense checkpoint found at {ckpt_dir}")
                continue

            if not replay_path.exists():
                print(f"  [SKIP] No replay buffer found at {replay_path}")
                continue

            # 1. Create environment
            env = make_env(task_name, seed=seed)
            obs_dim = env.observation_space.shape[0]
            act_dim = env.action_space.shape[0]
            act_low = env.action_space.low
            act_high = env.action_space.high

            # 2. Create agent
            sac_config = SACConfig(
                gamma=cfg["gamma"],
                tau=cfg["tau"],
                actor_lr=cfg["actor_lr"],
                critic_lr=cfg["critic_lr"],
                alpha_lr=cfg.get("alpha_lr", 3e-4),
                hidden_dims=tuple(cfg["hidden_dims"]),
                use_masking=False,  # Not needed for mask creation
            )

            set_global_seed(seed)
            agent = SACAgent(
                obs_dim=obs_dim,
                act_dim=act_dim,
                act_low=act_low,
                act_high=act_high,
                config=sac_config,
                seed=seed,
            )

            # 3. Load trained weights
            checkpointer = Checkpointer(ckpt_dir)
            restored_state = checkpointer.restore(agent.state, item="checkpoint_best.pkl")

            if restored_state is None:
                raise FileNotFoundError(f"Could not find checkpoint in {ckpt_dir}")

            print(f"    Loaded weights from step {restored_state.step}")
            agent.state = restored_state

            # 4. Load replay buffer
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

            # 5. Compute gradients by accumulating over multiple batches
            print(f"    Computing gradients using {num_gradient_batches} batches...")
            actor_grads, critic_grads = accumulate_gradient_statistics(
                agent=agent,
                replay_buffer=replay_buffer,
                num_batches=num_gradient_batches,
                batch_size=gradient_batch_size,
                normalize_obs=True,
            )

            # 6. Prune using gradient saliency
            print(f"    Pruning Actor with {gradient_method} saliency...")
            actor_mask = prune_kernels_by_gradient_saliency(
                params=agent.state.actor_params,
                gradients=actor_grads,
                target_sparsity=target_sparsity,
                method=gradient_method,
            )

            print(f"    Pruning Critic with {gradient_method} saliency...")
            critic_mask = prune_kernels_by_gradient_saliency(
                params=agent.state.critic_params,
                gradients=critic_grads,
                target_sparsity=target_sparsity,
                method=gradient_method,
            )

            # 7. Verify sparsity
            act_sp = compute_sparsity(actor_mask)
            crit_sp = compute_sparsity(critic_mask)
            print(f"    > Actual Actor Sparsity: {act_sp:.2%}")
            print(f"    > Actual Critic Sparsity: {crit_sp:.2%}")

            # 8. Save masks
            mask_data = {
                "actor": actor_mask,
                "critic": critic_mask,
                "sparsity_target": target_sparsity,
                "pruning_method": gradient_method,
                "num_gradient_batches": num_gradient_batches,
                "task": task_name,
                "seed": seed,
            }

            os.makedirs(os.path.dirname(mask_out_path), exist_ok=True)

            with open(mask_out_path, "wb") as f:
                pickle.dump(mask_data, f)

            print(f"  > Gradient-based mask saved to {mask_out_path}")
            print()

            # Clean up
            env.close()

    print("=" * 80)
    print("✓ All gradient-based masks created successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
