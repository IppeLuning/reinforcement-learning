from __future__ import annotations

import os
import pickle
from typing import Any, Dict, Optional

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
from src.utils.types import Mask


def create_mask(
    cfg: Dict[str, Any],
    task_name: str,
    seed: int,
    ckpt_dir: str,
    mask_out_path: str,
    target_sparsity_actor: float,
    target_sparsity_critic: float,
    pruning_method: str = "magnitude",
    prev_mask_path: Optional[str] = None,
) -> None:
    """Executes Step 2 of the LTH pipeline: Generates binary pruning masks.

    This function identifies the "Winning Ticket" by:
    1. Loading the high-performance weights from the previous training round.
    2. Analyzing parameter importance via magnitude or gradient-based saliency.
    3. Intersection with the previous mask to ensure "once pruned, always pruned."
    4. Saving a new binary mask PyTree for use in the next retraining round.

    Args:
        cfg: Global configuration dictionary.
        task_name: Name of the Meta-World task.
        seed: Random seed for environment initialization.
        ckpt_dir: Directory containing the trained checkpoints from the previous round.
        mask_out_path: File path where the resulting mask dictionary will be saved.
        target_sparsity_actor: Target fraction of actor weights to prune (0.0 to 1.0).
        target_sparsity_critic: Target fraction of critic weights to prune (0.0 to 1.0).
        pruning_method: The importance criterion to use ("magnitude" or "gradient").
        prev_mask_path: Optional path to a mask from a prior iterative pruning round.

    Raises:
        FileNotFoundError: If the required checkpoints or replay buffers are missing.
        ValueError: If an unsupported pruning method is specified.
    """

    if os.path.exists(mask_out_path):
        print(f"  [Skip] Mask already exists: {mask_out_path}")
        return

    print(
        f"  > Creating {pruning_method.upper()} Mask for {task_name} (Seed {seed})..."
    )
    print(
        f"    Target Actor: {target_sparsity_actor:.2%} | Target Critic: {target_sparsity_critic:.2%}"
    )

    # 2. Re-Initialize Agent (To get the correct State structure for loading)
    hp: Dict[str, Any] = cfg["hyperparameters"]
    defaults: Dict[str, Any] = hp["defaults"]
    task_overrides: Dict[str, Any] = hp.get("tasks", {}).get(task_name, {})
    params: Dict[str, Any] = {**defaults, **task_overrides}

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

    # 3. Load the Weights from Previous Round
    checkpointer = Checkpointer(ckpt_dir)
    restored_state = checkpointer.restore(agent.state, item="checkpoint_best.pkl")

    if restored_state is None:
        print(
            "    ! 'checkpoint_best.pkl' not found, falling back to 'checkpoint_final.pkl'"
        )
        restored_state = checkpointer.restore(agent.state, item="checkpoint_final.pkl")

    if restored_state is None:
        raise FileNotFoundError(f"Could not find any checkpoints in {ckpt_dir}")

    agent.state = restored_state

    # 4. Load Previous Mask (for Iterative Pruning)
    prev_actor_mask: Optional[Mask] = None
    prev_critic_mask: Optional[Mask] = None

    if prev_mask_path and os.path.exists(prev_mask_path):
        print(f"    Loading previous mask from: {prev_mask_path}")
        with open(prev_mask_path, "rb") as f:
            prev_data: Dict[str, Any] = pickle.load(f)
            prev_actor_mask = prev_data["actor"]
            prev_critic_mask = prev_data["critic"]

    # 5. Prune Parameters
    if pruning_method == "magnitude":
        print(f"    Pruning Actor to {target_sparsity_actor:.2%} (magnitude)...")
        actor_mask = prune_kernels_by_magnitude(
            restored_state.actor_params,
            target_sparsity_actor,
            prev_mask=prev_actor_mask,
        )

        print(f"    Pruning Critic to {target_sparsity_critic:.2%} (magnitude)...")
        critic_mask = prune_kernels_by_magnitude(
            restored_state.critic_params,
            target_sparsity_critic,
            prev_mask=prev_critic_mask,
        )

    elif pruning_method == "gradient":
        gradient_method: str = cfg.get("pruning", {}).get("gradient_method", "taylor")
        num_gradient_batches: int = cfg.get("pruning", {}).get(
            "num_gradient_batches", 100
        )
        gradient_batch_size: int = cfg.get("pruning", {}).get(
            "gradient_batch_size", 256
        )

        replay_path: str = os.path.join(ckpt_dir, "replay_buffer.pkl")
        if not os.path.exists(replay_path):
            raise FileNotFoundError(f"Replay buffer not found at {replay_path}.")

        with open(replay_path, "rb") as f:
            replay_data: Dict[str, Any] = pickle.load(f)

        replay_buffer = ReplayBuffer.create(
            obs_dim=obs_dim,
            act_dim=act_dim,
            capacity=replay_data["capacity"],
        )
        replay_buffer.load(replay_data)

        actor_grads, critic_grads = accumulate_gradient_statistics(
            agent=agent,
            replay_buffer=replay_buffer,
            num_batches=num_gradient_batches,
            batch_size=gradient_batch_size,
            normalize_obs=True,
        )

        print(
            f"    Pruning Actor to {target_sparsity_actor:.2%} ({gradient_method} saliency)..."
        )
        actor_mask = prune_kernels_by_gradient_saliency(
            params=restored_state.actor_params,
            gradients=actor_grads,
            target_sparsity=target_sparsity_actor,
            method=gradient_method,
            prev_mask=prev_actor_mask,
        )

        print(
            f"    Pruning Critic to {target_sparsity_critic:.2%} ({gradient_method} saliency)..."
        )
        critic_mask = prune_kernels_by_gradient_saliency(
            params=restored_state.critic_params,
            gradients=critic_grads,
            target_sparsity=target_sparsity_critic,
            method=gradient_method,
            prev_mask=prev_critic_mask,
        )
    else:
        raise ValueError(f"Unknown pruning_method: {pruning_method}")

    # 6. Verify Sparsity
    act_sp: float = compute_sparsity(actor_mask)
    crit_sp: float = compute_sparsity(critic_mask)
    print(f"    > Actual Actor Sparsity: {act_sp:.2%}")
    print(f"    > Actual Critic Sparsity: {crit_sp:.2%}")

    # 7. Save Masks
    mask_data: Dict[str, Any] = {
        "actor": actor_mask,
        "critic": critic_mask,
        "sparsity_target_actor": target_sparsity_actor,
        "sparsity_target_critic": target_sparsity_critic,
        "pruning_method": pruning_method,
        "task": task_name,
        "seed": seed,
    }

    os.makedirs(os.path.dirname(mask_out_path), exist_ok=True)
    with open(mask_out_path, "wb") as f:
        pickle.dump(mask_data, f)

    print(f"  > Mask saved to {mask_out_path}")
    env.close()
