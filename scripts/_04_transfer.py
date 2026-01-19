"""
Comparison Experiment Orchestrator.
Compares "Ticket Transfer" (Learned Mask + Rewound Weights) against
a "Random Baseline" (Random Mask + Random Weights).

Usage:
    python run_comparison.py
"""

import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np
import yaml

# Assuming train_mask is the sparse training loop
from scripts._03_train_ticket import train_mask

# Import pipeline components
from src.agents import SACAgent, SACConfig
from src.envs import make_metaworld_env
from src.utils import set_seed


def generate_random_mask(task, seed, sparsity, save_path, cfg):
    """
    Generates a random binary mask with specific sparsity (e.g., 0.8 = 80% zeros).
    """
    print(f"  [Gen] Generating {sparsity*100}% sparse random mask for {task}...")

    hp = cfg["hyperparameters"]
    env, obs_dim, act_dim, _, _ = make_metaworld_env(
        task, hp["max_episode_steps"], hp["defaults"]["scale_rewards"], seed
    )

    # Dummy agent to get shapes
    temp_config = SACConfig(hidden_dims=tuple(hp["hidden_dims"]))
    temp_agent = SACAgent(
        obs_dim, act_dim, np.zeros(act_dim), np.zeros(act_dim), temp_config, seed
    )

    rng = jax.random.PRNGKey(seed)

    def random_mask_leaf(leaf, key):
        # Create mask: 1 with prob (1-sparsity), 0 with prob sparsity
        # Using Bernoulli distribution
        mask = jax.random.bernoulli(key, p=(1.0 - sparsity), shape=leaf.shape)
        return mask.astype(jnp.float32)

    # Split keys for Actor
    actor_leaves, actor_def = jax.tree_util.tree_flatten(temp_agent.state.actor_params)
    rng, *actor_keys = jax.random.split(rng, len(actor_leaves) + 1)

    actor_mask_leaves = [
        random_mask_leaf(leaf, key) for leaf, key in zip(actor_leaves, actor_keys)
    ]
    actor_mask = jax.tree_util.tree_unflatten(actor_def, actor_mask_leaves)

    # Split keys for Critic
    critic_leaves, critic_def = jax.tree_util.tree_flatten(
        temp_agent.state.critic_params
    )
    rng, *critic_keys = jax.random.split(rng, len(critic_leaves) + 1)

    critic_mask_leaves = [
        random_mask_leaf(leaf, key) for leaf, key in zip(critic_leaves, critic_keys)
    ]
    critic_mask = jax.tree_util.tree_unflatten(critic_def, critic_mask_leaves)

    mask_data = {
        "actor": actor_mask,
        "critic": critic_mask,
        "sparsity_target": sparsity,
        "source_task": "random_initialization",
        "seed": seed,
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(mask_data, f)

    env.close()
    print(f"  [Save] Random mask saved to {save_path}")


def adapt_and_save_mask(source_mask_path, target_mask_path, target_task, seed, cfg):
    """Adapts source mask to target architecture (handling shape mismatches)"""
    print(f"  [Adapt] Adapting mask from {source_mask_path}...")

    with open(source_mask_path, "rb") as f:
        source_data = pickle.load(f)

    hp = cfg["hyperparameters"]
    env, obs_dim, act_dim, _, _ = make_metaworld_env(
        target_task, hp["max_episode_steps"], hp["defaults"]["scale_rewards"], seed
    )

    # Dummy agent for target structure
    temp_config = SACConfig(hidden_dims=tuple(hp["hidden_dims"]))
    temp_agent = SACAgent(
        obs_dim, act_dim, np.zeros(act_dim), np.zeros(act_dim), temp_config, seed
    )

    def adapt_leaf(source_leaf, target_shape):
        if source_leaf.shape != target_shape:
            # Fallback to Dense (all ones) on mismatch
            return jnp.ones(target_shape)
        return source_leaf

    # Adapt Actor
    src_flat, _ = jax.tree_util.tree_flatten(source_data["actor"])
    tgt_flat, tgt_def = jax.tree_util.tree_flatten(temp_agent.state.actor_params)
    new_actor = [adapt_leaf(s, t.shape) for s, t in zip(src_flat, tgt_flat)]
    actor_mask = jax.tree_util.tree_unflatten(tgt_def, new_actor)

    # Adapt Critic
    src_flat_c, _ = jax.tree_util.tree_flatten(source_data["critic"])
    tgt_flat_c, tgt_def_c = jax.tree_util.tree_flatten(temp_agent.state.critic_params)
    new_critic = [adapt_leaf(s, t.shape) for s, t in zip(src_flat_c, tgt_flat_c)]
    critic_mask = jax.tree_util.tree_unflatten(tgt_def_c, new_critic)

    mask_data = {"actor": actor_mask, "critic": critic_mask, "adapted_to": target_task}

    os.makedirs(os.path.dirname(target_mask_path), exist_ok=True)
    with open(target_mask_path, "wb") as f:
        pickle.dump(mask_data, f)
    env.close()


def main():
    # === CONFIGURATION ===
    SOURCE_TASK = "reach-v3"
    TARGET_TASK = "push-v3"
    SEED = 2000
    SOURCE_ROUND = 4

    # Comparison Flags
    RUN_TICKET_TRANSFER = False  # Rewound Weights + Reach Mask
    RUN_RANDOM_BASELINE = True  # Random Weights + Random Mask (80%)

    RANDOM_SPARSITY = 0.8
    # =====================

    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(SEED)
    base_exp_dir = f"data/experiments/{TARGET_TASK}/seed_{SEED}"

    # 1. Prepare Weight Checkpoints (Without full dense training)
    # ----------------------------------------------------------------
    # Rewind weights: The state of the network at step k (used for Ticket Transfer)
    rewind_steps = 20000
    rewind_ckpt_path = os.path.join(base_exp_dir, "round_0", "checkpoint_rewind.pkl")

    # Random Init weights: Pure initialization (Step 0) (used for Random Baseline)
    random_init_path = os.path.join(base_exp_dir, "round_0", "checkpoint_init.pkl")

    # 2. Experiment A: Ticket Transfer
    # ----------------------------------------------------------------
    if RUN_TICKET_TRANSFER:
        print(f"\n[Exp A] Running Ticket Transfer (Rewound Weights + Reach Mask)...")

        # Paths
        source_mask = (
            f"data/experiments/{SOURCE_TASK}/seed_{SEED}/round_{SOURCE_ROUND}/mask.pkl"
        )
        exp_dir = os.path.join(base_exp_dir, "transfer_reach_mask")
        target_mask = os.path.join(exp_dir, "mask_adapted.pkl")

        if os.path.exists(source_mask):
            adapt_and_save_mask(source_mask, target_mask, TARGET_TASK, SEED, cfg)

            # Train using REWOUND weights and ADAPTED mask
            train_mask(
                cfg,
                TARGET_TASK,
                SEED,
                mask_path=target_mask,
                save_dir=exp_dir,
                rewind_ckpt_path=rewind_ckpt_path,
            )
        else:
            print(f"Skipping Exp A: Source mask not found at {source_mask}")

    # 3. Experiment B: Random Baseline
    # ----------------------------------------------------------------
    if RUN_RANDOM_BASELINE:
        print(f"\n[Exp B] Running Random Baseline (Random Weights + Random Mask)...")

        exp_dir = os.path.join(
            base_exp_dir, f"random_sparse_{int(RANDOM_SPARSITY*100)}"
        )
        random_mask_path = os.path.join(exp_dir, "mask_random.pkl")

        # Generate Random Mask
        generate_random_mask(TARGET_TASK, SEED, RANDOM_SPARSITY, random_mask_path, cfg)

        # Train using RANDOM INIT weights and RANDOM mask
        # Note: We pass random_init_path to ensure we start from strictly seeded initialization
        train_mask(
            cfg,
            TARGET_TASK,
            SEED,
            mask_path=random_mask_path,
            save_dir=exp_dir,
            rewind_ckpt_path=random_init_path,
        )


if __name__ == "__main__":
    main()
