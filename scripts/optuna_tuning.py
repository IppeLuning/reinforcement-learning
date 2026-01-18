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

from src.agents import SACAgent, SACConfig
from src.data import ReplayBuffer
from src.envs import make_metaworld_env
from src.training.evaluation import evaluate
from src.utils import set_seed


def create_objective(task: str, tuning_steps: int = 100_000):
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

        # Setup
        seed = 42
        set_seed(seed)

        env, obs_dim, act_dim, act_low, act_high = make_metaworld_env(
            task, max_episode_steps=400, scale_factor=2, seed=seed
        )

        config = SACConfig(
            gamma=0.99,
            tau=tau,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            alpha_lr=alpha_lr,
            init_alpha=init_alpha,
            hidden_dims=actor_hidden_dims,  # fallback (not used with separate dims)
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

        # Training loop with pruning checkpoints
        obs, _ = env.reset(seed=seed)
        start_steps = 5000
        eval_interval = tuning_steps // 3  # 3 pruning checkpoints

        for step in range(1, tuning_steps + 1):
            # Action selection
            if step < start_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs, eval_mode=False)

            next_obs, reward, done, truncated, info = env.step(action)
            buffer.store(obs, action, reward, next_obs, float(done))
            obs = next_obs

            if done or truncated:
                obs, _ = env.reset()

            # Training
            if step >= start_steps and buffer.is_ready(batch_size):
                agent.update(buffer, batch_size)

            # Pruning checkpoint
            if step % eval_interval == 0:
                eval_metrics = evaluate(env, agent, num_episodes=5)
                mean_success = eval_metrics["mean_success"]

                trial.report(mean_success, step)
                if trial.should_prune():
                    env.close()
                    raise optuna.TrialPruned()

        # Final evaluation
        eval_metrics = evaluate(env, agent, num_episodes=10)
        env.close()

        return eval_metrics["mean_success"]

    return objective


def main():
    parser = argparse.ArgumentParser(description="Optuna tuning for SAC")
    parser.add_argument("--task", type=str, default="push-v3", help="MetaWorld task")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--tuning-steps", type=int, default=100_000, help="Steps per trial")
    args = parser.parse_args()

    # Storage for persistence (auto-resumes)
    os.makedirs(f"results/optuna/{args.task}", exist_ok=True)
    storage = f"sqlite:///results/optuna/{args.task}/study.db"

    study = optuna.create_study(
        study_name=f"sac_{args.task}",
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        storage=storage,
        load_if_exists=True,
    )

    study.optimize(
        create_objective(args.task, args.tuning_steps),
        n_trials=args.n_trials,
        n_jobs=2,  # M1 Max: 2 parallel trials leaves room for JAX Metal
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
