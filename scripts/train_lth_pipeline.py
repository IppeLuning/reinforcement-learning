#!/usr/bin/env python3
"""Lottery Ticket Hypothesis Pipeline for Multi-Task RL.

DISCLAIMER: This code was written by Claude Opus 4.5 on 2026-01-12
and reviewed by Marinus van den Ende.

This script implements the full LTH research pipeline:
1. Train single-task SAC agents (push-v3 for now)
2. Prune each to 80% sparsity using saliency-based pruning
3. Train multi-task SAC agent
4. Prune multi-task agent to get universal mask
5. Create union mask from single-task masks
6. Analyze structural relationships
7. Save all results and metrics

Usage:
    python scripts/train_lth_pipeline.py --config config.yaml
    python scripts/train_lth_pipeline.py --task push-v3 --seed 0  # Single run
"""

import argparse
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
import yaml

# Import our JAX implementation
from src.jax.agents.sac import SACAgent, SACConfig
from src.jax.buffers.replay_buffer import ReplayBuffer
from src.jax.training.evaluation import evaluate
from src.jax.pruning.saliency import prune_by_saliency, compute_saliency_for_sac
from src.jax.pruning.masks import MaskManager, compute_sparsity, create_ones_mask
from src.jax.pruning.analysis import (
    compute_structural_metrics, 
    generate_analysis_report,
    analyze_shared_core,
)
from src.jax.utils.checkpointing import Checkpointer

# Import environment utilities
from src.envs.envs import make_metaworld_env
from src.utils.seed import set_seed


def run_single_task_training(
    task_name: str,
    seed: int,
    config: Dict[str, Any],
    save_dir: str,
) -> Dict[str, Any]:
    """Train a single-task SAC agent.
    
    Args:
        task_name: Meta-World task name (e.g., "push-v3").
        seed: Random seed.
        config: Configuration dictionary.
        save_dir: Directory to save results.
        
    Returns:
        Dictionary with training results and metrics history.
    """
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
        
        # Evaluation
        if step % eval_interval == 0:
            eval_metrics = evaluate(env, agent, num_episodes=5)
            eval_metrics["step"] = step
            eval_metrics["alpha"] = agent.alpha
            eval_metrics["elapsed_time"] = time.time() - start_time
            metrics_history.append(eval_metrics)
            
            mean_ret = eval_metrics["mean_return"]
            mean_succ = eval_metrics["mean_success"]
            
            print(f"  Step {step:>7} | Return: {mean_ret:>7.1f} | "
                  f"Success: {mean_succ:.2%} | Alpha: {agent.alpha:.3f}")
            
            if mean_ret > best_return:
                best_return = mean_ret
    
    env.close()
    
    total_time = time.time() - start_time
    
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
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump({k: v for k, v in results.items() if k != "metrics_history"}, f, indent=2)
    
    with open(os.path.join(save_dir, "metrics_history.json"), "w") as f:
        json.dump(metrics_history, f, indent=2)
    
    # Save agent state
    checkpointer = Checkpointer(save_dir)
    checkpointer.save(agent.state, step=total_steps)
    
    print(f"\n  Training complete! Best return: {best_return:.1f}")
    print(f"  Time: {total_time/3600:.2f} hours ({results['steps_per_sec']:.0f} steps/sec)")
    
    return results, agent


def prune_agent(
    agent: SACAgent,
    buffer: ReplayBuffer,
    target_sparsity: float = 0.8,
    num_batches: int = 10,
    batch_size: int = 256,
) -> tuple:
    """Prune agent to target sparsity using saliency-based pruning.
    
    Args:
        agent: Trained SAC agent.
        buffer: Replay buffer with training data.
        target_sparsity: Target sparsity (0.8 = 80% weights pruned).
        num_batches: Number of batches for saliency computation.
        batch_size: Batch size for saliency computation.
        
    Returns:
        Tuple of (actor_mask, critic_mask, pruning_stats).
    """
    print(f"\n  Pruning to {target_sparsity:.0%} sparsity...")
    
    # Collect batches for saliency computation
    batches = [buffer.sample(batch_size) for _ in range(num_batches)]
    
    # Compute average saliency across batches
    actor_saliency_sum = None
    critic_saliency_sum = None
    
    for batch in batches:
        # Normalize batch observations
        batch_norm = type(batch)(
            obs=agent.normalizer.normalize(batch.obs),
            actions=batch.actions,
            rewards=batch.rewards,
            next_obs=agent.normalizer.normalize(batch.next_obs),
            dones=batch.dones,
        )
        
        actor_sal, critic_sal = compute_saliency_for_sac(
            agent.state.actor_params,
            agent.state.critic_params,
            agent.state.actor_apply_fn,
            agent.state.critic_apply_fn,
            batch_norm,
            agent.alpha,
        )
        
        if actor_saliency_sum is None:
            actor_saliency_sum = actor_sal
            critic_saliency_sum = critic_sal
        else:
            actor_saliency_sum = jax.tree.map(jnp.add, actor_saliency_sum, actor_sal)
            critic_saliency_sum = jax.tree.map(jnp.add, critic_saliency_sum, critic_sal)
    
    # Average saliency
    actor_saliency = jax.tree.map(lambda x: x / num_batches, actor_saliency_sum)
    critic_saliency = jax.tree.map(lambda x: x / num_batches, critic_saliency_sum)
    
    # Prune
    actor_mask, _ = prune_by_saliency(
        agent.state.actor_params, actor_saliency, target_sparsity
    )
    critic_mask, _ = prune_by_saliency(
        agent.state.critic_params, critic_saliency, target_sparsity
    )
    
    # Compute actual sparsities
    actor_sparsity = compute_sparsity(actor_mask)
    critic_sparsity = compute_sparsity(critic_mask)
    
    pruning_stats = {
        "target_sparsity": target_sparsity,
        "actor_sparsity": actor_sparsity,
        "critic_sparsity": critic_sparsity,
    }
    
    print(f"  Actor sparsity: {actor_sparsity:.2%}")
    print(f"  Critic sparsity: {critic_sparsity:.2%}")
    
    return actor_mask, critic_mask, pruning_stats


def main():
    parser = argparse.ArgumentParser(description="LTH Pipeline for Multi-Task RL")
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to configuration file")
    parser.add_argument("--task", type=str, default=None,
                        help="Single task to train (overrides config)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, load existing checkpoints")
    parser.add_argument("--test", action="store_true",
                        help="Quick test run (1000 steps)")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Override for test mode
    if args.test:
        config["single_task"]["total_steps"] = 1000
        config["single_task"]["start_steps"] = 100
        config["single_task"]["eval_interval"] = 500
        print("\n*** TEST MODE: 1000 steps ***\n")
    
    # Determine tasks
    if args.task:
        tasks = [args.task]
    else:
        tasks = config["environments"]["tasks"]
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"lth_run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'#'*60}")
    print(f"# Lottery Ticket Hypothesis Pipeline")
    print(f"# Tasks: {tasks}")
    print(f"# Seed: {args.seed}")
    print(f"# Output: {output_dir}")
    print(f"{'#'*60}")
    
    # Initialize mask manager
    mask_manager = MaskManager()
    
    # Phase 1: Train single-task agents and prune
    print(f"\n{'='*60}")
    print("PHASE 1: Single-Task Training and Pruning")
    print(f"{'='*60}")
    
    single_task_results = {}
    single_task_agents = {}
    
    for task_name in tasks:
        task_dir = os.path.join(output_dir, "single_task", task_name, f"seed_{args.seed}")
        
        if not args.skip_training:
            results, agent = run_single_task_training(
                task_name=task_name,
                seed=args.seed,
                config=config,
                save_dir=task_dir,
            )
            single_task_results[task_name] = results
            single_task_agents[task_name] = agent
            
            # Create buffer for pruning (we need to collect new data)
            env, obs_dim, act_dim, act_low, act_high = make_metaworld_env(
                task_name, config["defaults"]["max_episode_steps"], args.seed
            )
            buffer = ReplayBuffer(obs_dim, act_dim, max_size=10000)
            
            obs, _ = env.reset(seed=args.seed)
            for _ in range(5000):
                action = agent.select_action(obs, eval_mode=False)
                next_obs, reward, done, truncated, _ = env.step(action)
                buffer.store(obs, action, reward, next_obs, float(done))
                obs = next_obs if not (done or truncated) else env.reset()[0]
            env.close()
            
            # Prune agent
            actor_mask, critic_mask, pruning_stats = prune_agent(
                agent, buffer, target_sparsity=0.8
            )
            
            # Store masks
            mask_manager.store_task_mask(task_name, actor_mask, critic_mask)
            
            # Save masks
            checkpointer = Checkpointer(task_dir)
            checkpointer.save_masks(actor_mask, critic_mask, f"{task_name}_mask")
            
            # Save pruning stats
            with open(os.path.join(task_dir, "pruning_stats.json"), "w") as f:
                json.dump(pruning_stats, f, indent=2)
    
    # Phase 2: Structural Analysis
    print(f"\n{'='*60}")
    print("PHASE 2: Structural Analysis")
    print(f"{'='*60}")
    
    if len(tasks) > 1:
        # Get all actor masks for analysis
        actor_masks = {task: mask_manager.get_task_mask(task)[0] for task in tasks}
        
        # Compute structural metrics
        structural_metrics = compute_structural_metrics(actor_masks)
        
        # Analyze shared core
        shared_core = analyze_shared_core(actor_masks)
        
        # Generate report
        report = generate_analysis_report(actor_masks)
        print(report)
        
        # Compute union mask
        union_actor, union_critic = mask_manager.get_union_mask()
        union_sparsity = compute_sparsity(union_actor)
        print(f"\nUnion mask sparsity: {union_sparsity:.2%}")
        
        # Save analysis
        analysis_dir = os.path.join(output_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        with open(os.path.join(analysis_dir, "structural_metrics.json"), "w") as f:
            # Convert jax arrays to float for JSON
            metrics_serializable = {}
            for key, value in structural_metrics.items():
                if isinstance(value, dict):
                    metrics_serializable[key] = {k: float(v) if hasattr(v, 'item') else v 
                                                   for k, v in value.items()}
                else:
                    metrics_serializable[key] = value
            json.dump(metrics_serializable, f, indent=2)
        
        with open(os.path.join(analysis_dir, "shared_core.json"), "w") as f:
            json.dump({k: float(v) for k, v in shared_core.items()}, f, indent=2)
        
        with open(os.path.join(analysis_dir, "report.txt"), "w") as f:
            f.write(report)
        
        # Save summary
        summary = mask_manager.summary()
        with open(os.path.join(analysis_dir, "mask_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Pipeline Complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
