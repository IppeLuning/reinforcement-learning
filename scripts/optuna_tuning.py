"""Optuna hyperparameter tuning for SAC agent.

Optimized for M1 Max with 32GB RAM.
Usage: python scripts/optuna_tuning.py --task push-v3 --n-trials 50
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os
from typing import Any, Dict

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

import numpy as np

from src.agents import SACAgent, SACConfig
from src.data import ReplayBuffer
from src.envs import make_metaworld_env, make_vectorized_metaworld_env
from src.training.evaluation import evaluate
from src.utils import set_seed


def create_objective(task: str, tuning_steps: int = 100_000, num_envs: int = 8):
    """Create Optuna objective function for a specific task."""

    def objective(trial: optuna.Trial) -> float:
        # Actor hyperparameters (separate)
        actor_lr = trial.suggest_float("actor_lr", 1e-5, 1e-3, log=True)
        actor_hidden = trial.suggest_categorical(
            "actor_hidden", ["128,128", "256,256", "256,128", "128,128,128"]
        )
        actor_hidden_dims = tuple(map(int, actor_hidden.split(",")))

        # Critic hyperparameters (shared for Q1/Q2)
        critic_lr = trial.suggest_float("critic_lr", 1e-5, 1e-3, log=True)
        critic_hidden = trial.suggest_categorical(
            "critic_hidden", ["128,128", "256,256", "256,128", "128,128,128"]
        )
        critic_hidden_dims = tuple(map(int, critic_hidden.split(",")))

        # SAC hyperparameters
        alpha_lr = trial.suggest_float("alpha_lr", 1e-6, 1e-4, log=True)
        tau = trial.suggest_float("tau", 0.001, 0.02)
        init_alpha = trial.suggest_float("init_alpha", 0.05, 1.0, log=True)

        # Training hyperparameters
        batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048, 4096])

        print(f"\n{'='*60}", flush=True)
        print(f"Trial {trial.number}: actor_lr={actor_lr:.2e}, critic_lr={critic_lr:.2e}", flush=True)
        print(f"  actor={actor_hidden}, critic={critic_hidden}, batch={batch_size}", flush=True)
        print(f"{'='*60}", flush=True)

        # Setup
        seed = 42
        set_seed(seed)

        # Vectorized env for fast data collection
        vec_env, obs_dim, act_dim, act_low, act_high = make_vectorized_metaworld_env(
            task, max_episode_steps=400, scale_factor=2,
            num_envs=num_envs, strategy="sync", base_seed=seed
        )
        # Single env for evaluation
        eval_env, _, _, _, _ = make_metaworld_env(task, 400, 2, seed)

        config = SACConfig(
            gamma=0.99,
            tau=tau,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            alpha_lr=alpha_lr,
            init_alpha=init_alpha,
            hidden_dims=actor_hidden_dims,
        )

        agent = SACAgent(obs_dim, act_dim, act_low, act_high, config, seed)
        # Reinitialize with separate hidden dims
        from src.training.train_state import create_sac_train_state
        import jax

        agent.state = create_sac_train_state(
            key=jax.random.PRNGKey(seed),
            obs_dim=obs_dim,
            act_dim=act_dim,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            alpha_lr=alpha_lr,
            init_alpha=init_alpha,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
        )

        buffer = ReplayBuffer(obs_dim, act_dim, max_size=500_000)

        # Training loop with vectorized collection
        obs, _ = vec_env.reset(seed=seed)
        start_steps = 2000
        eval_interval = tuning_steps // 5
        log_interval = 5000
        total_env_steps = 0
        
        print(f"  Starting training with {num_envs} parallel envs...", flush=True)

        while total_env_steps < tuning_steps:
            # Action selection for all envs
            if total_env_steps < start_steps:
                actions = np.array([vec_env.action_space.sample() for _ in range(num_envs)])
            else:
                actions = np.array([agent.select_action(obs[i], eval_mode=False) for i in range(num_envs)])

            # Step all envs
            next_obs, rewards, dones, truncateds, infos = vec_env.step(actions)
            
            # Store transitions
            for i in range(num_envs):
                buffer.store(obs[i], actions[i], rewards[i], next_obs[i], float(dones[i]))
            
            obs = next_obs
            total_env_steps += num_envs

            # Training (2 updates per env step for better sample efficiency)
            if total_env_steps >= start_steps and buffer.is_ready(batch_size):
                for _ in range(2):
                    agent.update(buffer, batch_size)

            # Progress logging
            if total_env_steps % log_interval < num_envs:
                print(f"  Trial {trial.number} | Step {total_env_steps}/{tuning_steps} ({100*total_env_steps/tuning_steps:.0f}%)", flush=True)

            # Pruning checkpoint
            if total_env_steps % eval_interval < num_envs:
                eval_metrics = evaluate(eval_env, agent, num_episodes=5)
                mean_success = eval_metrics["mean_success"]
                print(f"  [Eval] Trial {trial.number} | Step {total_env_steps} | Success: {mean_success:.2%}", flush=True)

                trial.report(mean_success, total_env_steps)
                if trial.should_prune():
                    print(f"  Trial {trial.number} PRUNED at step {total_env_steps}", flush=True)
                    vec_env.close()
                    eval_env.close()
                    raise optuna.TrialPruned()

        # Final evaluation
        eval_metrics = evaluate(eval_env, agent, num_episodes=10)
        vec_env.close()
        eval_env.close()
        
        final_success = eval_metrics["mean_success"]
        print(f"  Trial {trial.number} COMPLETE | Final success: {final_success:.2%}", flush=True)
        return final_success

    return objective


def main():
    parser = argparse.ArgumentParser(description="Optuna tuning for SAC")
    parser.add_argument("--task", type=str, default="push-v3", help="MetaWorld task")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--tuning-steps", type=int, default=100_000, help="Steps per trial")
    parser.add_argument("--num-envs", type=int, default=8, help="Parallel envs (8 for M1 Max)")
    parser.add_argument("--study-name", type=str, default="study", help="Name of the study")
    args = parser.parse_args()

    # Storage for persistence (auto-resumes)
    os.makedirs(f"results/optuna/{args.task}", exist_ok=True)
    storage = f"sqlite:///results/optuna/{args.task}/{args.study_name}.db"

    study = optuna.create_study(
        study_name=f"sac_{args.task}",
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        storage=storage,
        load_if_exists=True,
    )

    study.optimize(
        create_objective(args.task, args.tuning_steps, args.num_envs),
        n_trials=args.n_trials,
        n_jobs=1,  # Sequential for cleaner output (parallel can cause JAX issues)
        show_progress_bar=True,
    )

    print("\n" + "=" * 50)
    print("Best trial:")
    print(f"  Success rate: {study.best_trial.value:.2%}")
    print("  Params:")
    for k, v in study.best_trial.params.items():
        print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
