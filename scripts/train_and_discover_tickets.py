import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from src.agents.sac import SACAgent, SACConfig
from src.data.replay_buffer import ReplayBuffer
from src.envs.factory import make_metaworld_env
from src.training.evaluation import evaluate
from src.utils.checkpointing import Checkpointer
from src.utils.seed import set_seed


def prune_by_magnitude(params, target_sparsity=0.8):
    """
    Standard LTH Pruning: Prunes the smallest weights globally.

    Args:
        params: The trained PyTree of weights.
        target_sparsity: Float (e.g., 0.8 means remove 80% of weights).
    Returns:
        mask: Binary mask (1.0 = keep, 0.0 = kill).
    """
    # 1. Flatten all weights into one big array (Global Pruning)
    flat_params, tree_def = jax.tree.flatten(params)
    # Concatenate all leaves, taking absolute value
    all_weights = jnp.concatenate([jnp.abs(p).flatten() for p in flat_params])

    # 2. Determine the threshold value
    k = int(len(all_weights) * target_sparsity)
    # Sort and pick the k-th smallest value
    threshold = jnp.sort(all_weights)[k]

    # 3. Create mask: 1 if |w| > threshold, else 0
    def create_mask(w):
        return (jnp.abs(w) > threshold).astype(jnp.float32)

    mask = jax.tree.map(create_mask, params)

    # Verification
    total_params = len(all_weights)
    kept_params = jnp.sum(jnp.concatenate([m.flatten() for m in jax.tree.leaves(mask)]))
    actual_sparsity = 1.0 - (kept_params / total_params)
    print(f"  > Pruning Complete. Threshold: {threshold:.6f}")
    print(f"  > Target: {target_sparsity:.2%} | Actual: {actual_sparsity:.2%}")

    return mask


def run_ticket_discovery(task_name: str, seed: int, config: dict, output_dir: str):
    """
    Step 1 of LTH:
    1. Initialize Network (W0) -> SAVE IT
    2. Train to completion (W_final)
    3. Prune (Mask) -> SAVE IT
    """
    print(f"\n=== Starting Ticket Discovery: {task_name} (Seed {seed}) ===")

    set_seed(seed)

    # Extract config
    single_cfg = config["single_task"]
    defaults = single_cfg.get("defaults", {})
    task_overrides = single_cfg.get("tasks", {}).get(task_name, {})
    params = {**defaults, **task_overrides}
    params["hidden_dims"] = tuple(config["network"]["hidden_dims"])

    # Create environment
    env, obs_dim, act_dim, act_low, act_high = make_metaworld_env(
        task_name,
        config["defaults"]["max_episode_steps"],
        seed,
    )

    # Create agent
    sac_config = SACConfig(
        gamma=params.get("gamma", 0.99),
        tau=params.get("tau", 0.005),
        actor_lr=params.get("actor_lr", 3e-4),
        critic_lr=params.get("critic_lr", 3e-4),
        alpha_lr=params.get("alpha_lr", 3e-4),
        target_entropy_scale=params.get("target_entropy_scale", 1.0),
        auto_alpha=params.get("auto_alpha", True),
        init_alpha=params.get("init_alpha", 1.0),
        hidden_dims=params["hidden_dims"],
    )

    agent = SACAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        act_low=act_low,
        act_high=act_high,
        config=sac_config,
        seed=seed,
    )

    # 2. Infrastructure
    checkpointer = Checkpointer(output_dir)
    
    # Get the step at which to save initialization
    save_init_at_step = single_cfg.get("save_init_at_step", 0)
    
    # CRITICAL: Save Initial Weights (W0)
    # Save before training starts if save_init_at_step is 0
    if save_init_at_step == 0:
        print(f"  > Saving Initial Weights (W0) at step 0...")
        checkpointer.save(agent.state, filename="checkpoint_init")

    # Create replay buffer
    buffer = ReplayBuffer(
        obs_dim=obs_dim,
        act_dim=act_dim,
        max_size=single_cfg.get("replay_buffer_size", 1_000_000),
    )

    # Training parameters
    total_steps = single_cfg["total_steps"]
    start_steps = single_cfg["start_steps"]
    batch_size = single_cfg["batch_size"]
    eval_interval = single_cfg["eval_interval"]
    updates_per_step = params.get("updates_per_step", 1)

    print(f"\n{'='*60}")
    print(f"[Single-Task Training] {task_name} | Seed {seed}")
    print(f"{'='*60}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Hidden dims: {params['hidden_dims']}")
    print(f"  Init alpha: {sac_config.init_alpha}")

    # Training loop
    obs, _ = env.reset(seed=seed)
    episode_return = 0.0
    episode_success = 0.0
    episode_len = 0

    metrics_history = []
    best_return = float("-inf")
    start_time = time.time()

    for step in range(1, total_steps + 1):
        # Select action
        if step < start_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, eval_mode=False)

        # Environment step
        next_obs, reward, done, truncated, info = env.step(action)
        terminal = done or truncated

        # Store transition
        buffer.store(obs, action, reward, next_obs, float(done))

        # Update tracking
        obs = next_obs
        episode_return += reward
        episode_success = max(episode_success, float(info.get("success", 0.0)))
        episode_len += 1

        # Handle episode end
        if terminal:
            obs, _ = env.reset()
            episode_return = 0.0
            episode_success = 0.0
            episode_len = 0

        # Training updates
        if step >= start_steps and buffer.is_ready(batch_size):
            for _ in range(updates_per_step):
                agent.update(buffer, batch_size)

        # Save Initial Weights at specified step (if not saved at step 0)
        if save_init_at_step > 0 and step == save_init_at_step:
            print(f"  > Saving Initial Weights (W0) at step {step}...")
            checkpointer.save(agent.state, filename="checkpoint_init")

        # Evaluation
        if step % eval_interval == 0:
            eval_metrics = evaluate(env, agent, num_episodes=5)
            eval_metrics["step"] = step
            eval_metrics["alpha"] = agent.alpha
            eval_metrics["elapsed_time"] = time.time() - start_time
            metrics_history.append(eval_metrics)

            mean_ret = eval_metrics["mean_return"]
            mean_succ = eval_metrics["mean_success"]

            print(
                f"  Step {step:>7} | Return: {mean_ret:>7.1f} | "
                f"Success: {mean_succ:.2%} | Alpha: {agent.alpha:.3f}"
            )

            if mean_ret > best_return:
                best_return = mean_ret

    env.close()

    total_time = time.time() - start_time
    print(f"  > Training Complete in {total_time:.0f}s")

    results = {
        "task": task_name,
        "seed": seed,
        "total_steps": total_steps,
        "best_return": best_return,
        "final_return": metrics_history[-1]["mean_return"] if metrics_history else 0,
        "final_success": metrics_history[-1]["mean_success"] if metrics_history else 0,
        "wall_time_sec": total_time,
        "steps_per_sec": total_steps / total_time,
        "metrics_history": metrics_history,
    }

    # Save results
    save_dir = "results/ticket_discovery"
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(
            {k: v for k, v in results.items() if k != "metrics_history"}, f, indent=2
        )

    with open(os.path.join(save_dir, "metrics_history.json"), "w") as f:
        json.dump(metrics_history, f, indent=2)

    # Save agent state
    checkpointer = Checkpointer(save_dir)
    checkpointer.save(agent.state, filename="checkpoint_final")

    print(f"\n  Training complete! Best return: {best_return:.1f}")
    print(
        f"  Time: {total_time/3600:.2f} hours ({results['steps_per_sec']:.0f} steps/sec)"
    )

    print(f"  > Pruning Actor and Critic to 80% sparsity...")

    actor_mask = prune_by_magnitude(agent.state.actor_params, target_sparsity=0.8)
    critic_mask = prune_by_magnitude(agent.state.critic_params, target_sparsity=0.8)

    # 6. Save the Mask
    # Now we have everything: W0 (saved at step 0) and Mask (saved now).
    checkpointer.save_masks(
        actor_mask, critic_mask, filename=f"mask_{task_name}_s{seed}"
    )

    print(f"=== Ticket Discovery Complete ===")
    print(f"Saved: W0, W_final, and Masks in {output_dir}")

    return results, agent
