from __future__ import annotations

import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from scripts._03_train_ticket import train_mask
from src.agents import SACAgent, SACConfig
from src.envs import make_metaworld_env
from src.utils import set_seed
from src.utils.types import Mask, PRNGKey


def generate_random_mask(
    task: str, seed: int, sparsity: float, save_path: str, cfg: Dict[str, Any]
) -> None:
    """Generates a random binary mask using a Bernoulli distribution.

    Args:
        task: Name of the Meta-World task to define network shapes.
        seed: Random seed for reproducibility.
        sparsity: Fraction of weights to set to zero (e.g., 0.8 = 80% sparse).
        save_path: Path to save the generated mask pickle file.
        cfg: Configuration dictionary containing architecture hyperparameters.
    """
    print(f"  [Gen] Generating {sparsity*100}% sparse random mask for {task}...")

    hp: Dict[str, Any] = cfg["hyperparameters"]
    env, obs_dim, act_dim, _, _ = make_metaworld_env(
        task, hp["max_episode_steps"], hp["defaults"]["scale_rewards"], seed
    )

    temp_config = SACConfig(hidden_dims=tuple(hp["hidden_dims"]))
    temp_agent = SACAgent(
        obs_dim, act_dim, np.zeros(act_dim), np.zeros(act_dim), temp_config, seed
    )

    rng: PRNGKey = jax.random.PRNGKey(seed)

    def random_mask_leaf(leaf: jax.Array, key: PRNGKey) -> jax.Array:
        """Creates a binary mask leaf where 1 has probability (1 - sparsity)."""
        mask = jax.random.bernoulli(key, p=(1.0 - sparsity), shape=leaf.shape)
        return mask.astype(jnp.float32)

    # Actor Mask Generation
    actor_leaves, actor_def = jax.tree_util.tree_flatten(temp_agent.state.actor_params)
    rng, *actor_keys = jax.random.split(rng, len(actor_leaves) + 1)
    actor_mask_leaves = [
        random_mask_leaf(leaf, key) for leaf, key in zip(actor_leaves, actor_keys)
    ]
    actor_mask: Mask = jax.tree_util.tree_unflatten(actor_def, actor_mask_leaves)

    # Critic Mask Generation
    critic_leaves, critic_def = jax.tree_util.tree_flatten(
        temp_agent.state.critic_params
    )
    rng, *critic_keys = jax.random.split(rng, len(critic_leaves) + 1)
    critic_mask_leaves = [
        random_mask_leaf(leaf, key) for leaf, key in zip(critic_leaves, critic_keys)
    ]
    critic_mask: Mask = jax.tree_util.tree_unflatten(critic_def, critic_mask_leaves)

    mask_data: Dict[str, Any] = {
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


def adapt_and_save_mask(
    source_mask_path: str,
    target_mask_path: str,
    target_task: str,
    seed: int,
    cfg: Dict[str, Any],
) -> None:
    """Adapts a mask from a source task to a target architecture.

    If a leaf shape mismatch is detected (e.g., due to different observation
    dimensions), the function falls back to a dense (all ones) leaf for that
    specific parameter to prevent crashes.

    Args:
        source_mask_path: Path to the existing source mask.
        target_mask_path: Path to save the adapted mask.
        target_task: Task name for the target architecture.
        seed: Random seed.
        cfg: Configuration dictionary.
    """
    print(f"  [Adapt] Adapting mask from {source_mask_path}...")

    with open(source_mask_path, "rb") as f:
        source_data: Dict[str, Any] = pickle.load(f)

    hp: Dict[str, Any] = cfg["hyperparameters"]
    env, obs_dim, act_dim, _, _ = make_metaworld_env(
        target_task, hp["max_episode_steps"], hp["defaults"]["scale_rewards"], seed
    )

    temp_config = SACConfig(hidden_dims=tuple(hp["hidden_dims"]))
    temp_agent = SACAgent(
        obs_dim, act_dim, np.zeros(act_dim), np.zeros(act_dim), temp_config, seed
    )

    def adapt_leaf(source_leaf: jax.Array, target_shape: Tuple[int, ...]) -> jax.Array:
        if source_leaf.shape != target_shape:
            return jnp.ones(target_shape)
        return source_leaf

    # Adapt Actor Mask
    src_flat, _ = jax.tree_util.tree_flatten(source_data["actor"])
    tgt_flat, tgt_def = jax.tree_util.tree_flatten(temp_agent.state.actor_params)
    new_actor = [adapt_leaf(s, t.shape) for s, t in zip(src_flat, tgt_flat)]
    actor_mask: Mask = jax.tree_util.tree_unflatten(tgt_def, new_actor)

    # Adapt Critic Mask
    src_flat_c, _ = jax.tree_util.tree_flatten(source_data["critic"])
    tgt_flat_c, tgt_def_c = jax.tree_util.tree_flatten(temp_agent.state.critic_params)
    new_critic = [adapt_leaf(s, t.shape) for s, t in zip(src_flat_c, tgt_flat_c)]
    critic_mask: Mask = jax.tree_util.tree_unflatten(tgt_def_c, new_critic)

    mask_data: Dict[str, Any] = {
        "actor": actor_mask,
        "critic": critic_mask,
        "adapted_to": target_task,
    }

    os.makedirs(os.path.dirname(target_mask_path), exist_ok=True)
    with open(target_mask_path, "wb") as f:
        pickle.dump(mask_data, f)
    env.close()


def main() -> None:
    """Main execution entry point for transfer and baseline experiments."""
    SOURCE_TASK: str = "reach-v3"
    TARGET_TASK: str = "push-v3"
    SEED: int = 2000
    SOURCE_ROUND: int = 4

    RUN_TICKET_TRANSFER: bool = True
    RUN_RANDOM_BASELINE: bool = False
    RANDOM_SPARSITY: float = 0.8

    with open("config.yaml", "r") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    set_seed(SEED)
    base_exp_dir: str = f"data/experiments/{TARGET_TASK}/seed_{SEED}"

    # 1. Prepare Weight Checkpoints
    rewind_ckpt_path: str = os.path.join(
        f"data/experiments/{SOURCE_TASK}/seed_{SEED}",
        "round_0",
        "checkpoint_rewind.pkl",
    )
    random_init_path: str = os.path.join(base_exp_dir, "round_0", "checkpoint_init.pkl")

    # 2. Experiment A: Ticket Transfer
    if RUN_TICKET_TRANSFER:
        print(
            f"\n[Exp A] Running Ticket Transfer (Rewound Weights + {SOURCE_TASK} Mask)..."
        )

        source_mask_path = (
            f"data/experiments/{SOURCE_TASK}/seed_{SEED}/round_{SOURCE_ROUND}/mask.pkl"
        )
        exp_dir = os.path.join(base_exp_dir, "transfer_reach_mask")
        target_mask_path = os.path.join(exp_dir, "mask_adapted.pkl")

        if os.path.exists(source_mask_path):
            adapt_and_save_mask(
                source_mask_path, target_mask_path, TARGET_TASK, SEED, cfg
            )

            train_mask(
                cfg,
                TARGET_TASK,
                SEED,
                mask_path=target_mask_path,
                save_dir=exp_dir,
                rewind_ckpt_path=rewind_ckpt_path,
            )
        else:
            print(f"Skipping Exp A: Source mask not found at {source_mask_path}")

    # 3. Experiment B: Random Baseline
    if RUN_RANDOM_BASELINE:
        print(f"\n[Exp B] Running Random Baseline (Random Weights + Random Mask)...")

        exp_dir = os.path.join(
            base_exp_dir, f"random_sparse_{int(RANDOM_SPARSITY*100)}"
        )
        random_mask_path = os.path.join(exp_dir, "mask_random.pkl")

        generate_random_mask(TARGET_TASK, SEED, RANDOM_SPARSITY, random_mask_path, cfg)

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
