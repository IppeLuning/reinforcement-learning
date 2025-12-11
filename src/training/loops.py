import json
import math
import os
import time

import numpy as np
import torch

from src.envs.envs import make_metaworld_env
from src.envs.multitask_wrapper import MultiTaskEnv
from src.models.multi_task_sac import MTSACAgent
from src.models.sac import SACAgent, SACConfig
from src.replay_buffer import ReplayBuffer
from src.utils.color import Color
from src.utils.device import get_device
from src.utils.seed import set_seed


def evaluate(env, agent, episodes: int = 5):
    """
    Returns dictionary of metrics:
      - mean_return
      - mean_success
      - mean_time_to_success (steps)
      - mean_smoothness (action jerk)
    """
    returns = []
    successes = []
    times_to_success = []
    action_smoothness = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_ret = 0.0

        # Track success timing
        ep_success_bool = False
        first_success_step = env._max_episode_steps

        # Track action smoothness
        episode_actions = []

        step_count = 0
        while not (done or truncated):
            act = agent.select_action(obs, eval_mode=True)
            episode_actions.append(act)

            obs, rew, done, truncated, info = env.step(act)
            ep_ret += rew
            step_count += 1

            # Check success (Meta-World specific)
            curr_success = float(info.get("success", 0.0))
            if curr_success > 0.0 and not ep_success_bool:
                ep_success_bool = True
                first_success_step = step_count

        # Post-Episode metrics
        returns.append(ep_ret)
        successes.append(1.0 if ep_success_bool else 0.0)
        times_to_success.append(first_success_step)

        # Smoothness: mean squared diff between consecutive actions
        if len(episode_actions) > 1:
            actions_arr = np.array(episode_actions)
            diffs = np.diff(actions_arr, axis=0)
            smoothness_score = np.mean(np.linalg.norm(diffs, axis=1) ** 2)
            action_smoothness.append(smoothness_score)

    metrics = {
        "mean_return": float(np.mean(returns)),
        "best_return": float(np.max(returns)),
        "mean_success": float(np.mean(successes)),
        "best_success": float(np.max(successes)),
        "mean_time_to_success": float(np.mean(times_to_success)),
        "mean_smoothness": (
            float(np.mean(action_smoothness)) if action_smoothness else 0.0
        ),
    }
    return metrics


# --- Core Training Loop ---
def run_training_loop(
    env,
    agent,
    replay_buffer,
    total_steps: int,
    start_steps: int,
    batch_size: int,
    eval_interval: int,
    save_dir: str,
    seed: int,
    task_name: str,
    target_mean_success: float,
    patience: int,
    algo: str,
    updates_per_step: int = 1,
    is_multitask: bool = False,
):
    start_time = time.time()
    obs, _ = env.reset(seed=seed)

    # Episode tracking
    episode_return = 0.0
    epsiode_success = 0.0
    episode_len = 0

    # Best stats tracking
    best_return_overall = -np.inf
    best_success_overall = 0.0
    best_step = 0

    # Early stopping tracking
    no_improve = 0
    success_patience = 0
    stop_reason = "max_steps"

    os.makedirs(save_dir, exist_ok=True)
    eval_episodes = 10 if is_multitask else 5

    for t in range(1, total_steps + 1):
        if t < start_steps:
            act = env.action_space.sample()
        else:
            act = agent.select_action(obs, eval_mode=False)

        next_obs, rew, done, truncated, info = env.step(act)

        # terminal is used only for episode control
        terminal = done or truncated

        # IMPORTANT: store only true termination (done) for bootstrapping mask
        done_float = float(done)

        replay_buffer.store(obs, act, rew, next_obs, done_float)
        obs = next_obs
        episode_return += rew
        epsiode_success = max(epsiode_success, float(info.get("success", 0.0)))
        episode_len += 1

        if terminal:
            c = Color.GREEN if epsiode_success >= 1.0 else Color.RED
            print(
                f"[Seed {seed}] Step {t} | "
                f"Task: {info.get('task_name', task_name)} | "
                f"Return: {episode_return:.2f} | "
                f"{c}Success: {epsiode_success:.2f}{Color.END} | "
                f"Len: {episode_len}"
            )
            obs, _ = env.reset()
            episode_return = 0.0
            epsiode_success = 0.0
            episode_len = 0

        if t >= start_steps:
            for _ in range(updates_per_step):
                agent.update(replay_buffer, batch_size)

        if t % eval_interval == 0:
            eval_metrics = evaluate(env, agent, episodes=eval_episodes)

            mean_ret = eval_metrics["mean_return"]
            best_ret = eval_metrics["best_return"]
            mean_succ = eval_metrics["mean_success"]
            best_succ = eval_metrics["best_success"]

            print(
                f"[Eval] Step {t} | Task: {task_name} | "
                f"Return: {mean_ret:>7.1f} (Best: {best_ret:>7.1f}) | "
                f"Success: {mean_succ:>4.2f} (Best: {best_succ:>4.2f}) | "
                f"Time-to-Succ: {eval_metrics['mean_time_to_success']:.1f} | "
                f"Smoothness: {eval_metrics['mean_smoothness']:.4f} | "
                f"No Improvement Steps: {no_improve}"
            )

            # --- Saving Logic ---
            if mean_ret > best_return_overall + 1e-3:
                best_return_overall = mean_ret
                best_success_overall = mean_succ
                best_step = t
                no_improve = 0

                actor_path = os.path.join(
                    save_dir,
                    f"{algo.lower()}_{task_name}_seed{seed}_best_actor.pt",
                )
                torch.save(agent.actor.state_dict(), actor_path)
                print(f"Saved new best actor (Return: {best_return_overall:.2f})")
            else:
                no_improve += 1

            # --- Early Stopping Logic ---
            if target_mean_success is not None and mean_succ >= target_mean_success:
                success_patience += 1
            else:
                success_patience = 0

            if success_patience >= patience:
                print(
                    f"[Seed {seed}] Early stopping triggered at step {t} (Success: {mean_succ:.2f})"
                )
                stop_reason = "early_stopping"
                break

    total_time = time.time() - start_time
    training_stats = {
        "task": task_name,
        "seed": seed,
        "algo": algo,
        "total_steps_trained": t,
        "stop_reason": stop_reason,
        "best_step": best_step,
        "best_return": best_return_overall,
        "best_success": best_success_overall,
        "wall_time_sec": total_time,
        "steps_per_sec": t / total_time if total_time > 0 else 0,
    }

    return training_stats


def _get_params_for_task(cfg, task_name):
    """
    Merges default settings with task-specific overrides.
    Returns:
      sac_params (dict): For SACConfig
      loop_params (dict): For training loop (patience, target_success)
    """
    raw_params = cfg["single_task"]["defaults"].copy()
    task_overrides = cfg["single_task"].get("tasks", {}).get(task_name, {})
    raw_params.update(task_overrides)

    raw_params["hidden_dims"] = tuple(cfg["network"]["hidden_dims"])

    target_mean_success = raw_params.pop("target_mean_success", None)
    patience = raw_params.pop("patience", math.inf)
    updates_per_step = raw_params.pop("updates_per_step", 1)

    return raw_params, target_mean_success, patience, updates_per_step


def train_single_task_session(cfg, task_name, seed):
    set_seed(seed)
    device = get_device()
    algo = cfg["single_task"]["algorithm"]

    sac_params_dict, target_success, patience, updates_per_step = _get_params_for_task(
        cfg, task_name
    )
    sac_config = SACConfig(**sac_params_dict)

    save_path = os.path.join(
        cfg["defaults"]["save_dir"], "single_task", task_name, f"seed_{seed}"
    )
    os.makedirs(save_path, exist_ok=True)

    env, obs_dim, act_dim, low, high = make_metaworld_env(
        task_name, cfg["defaults"]["max_episode_steps"], seed
    )

    agent = SACAgent(obs_dim, act_dim, low, high, device, sac_config)
    rb = ReplayBuffer(obs_dim, act_dim, cfg["single_task"]["replay_buffer_size"])

    print(f"\n--- [Single Task] {task_name} | Seed {seed} ---")
    print(
        f"    Params: Alpha={sac_config.init_alpha}, "
        f"Target Entropy Scale={sac_config.target_entropy_scale}, "
        f"Patience={patience}, Target={target_success}"
    )

    stats = run_training_loop(
        env,
        agent,
        rb,
        cfg["single_task"]["total_steps"],
        cfg["single_task"]["start_steps"],
        cfg["single_task"]["batch_size"],
        cfg["single_task"]["eval_interval"],
        save_path,
        seed,
        task_name,
        target_success,
        patience,
        algo,
        updates_per_step,
        is_multitask=False,
    )

    json_path = os.path.join(save_path, "results.json")
    full_log = {"config": cfg, "final_sac_params": sac_params_dict, "results": stats}
    with open(json_path, "w") as f:
        json.dump(full_log, f, indent=4)
    print(f"Stats saved to {json_path}")

    env.close()


def train_multitask_session(cfg, seed):
    set_seed(seed)
    device = get_device()
    algo = cfg["multi_task"]["algorithm"]
    task_name_for_log = "MT_shared"

    raw_params = cfg["multi_task"]["sac_params"].copy()
    raw_params["hidden_dims"] = tuple(cfg["network"]["hidden_dims"])

    target_success = raw_params.pop("target_mean_success", None)
    patience = raw_params.pop("patience", math.inf)

    sac_config = SACConfig(**raw_params)

    save_path = os.path.join(cfg["defaults"]["save_dir"], "multitask", f"seed_{seed}")
    os.makedirs(save_path, exist_ok=True)

    tasks = cfg["environments"]["tasks"]
    envs_dict = {}
    obs_dim, act_dim, low, high = 0, 0, 0, 0

    for t_name in tasks:
        e, o, a, l, h = make_metaworld_env(
            t_name, cfg["defaults"]["max_episode_steps"], seed
        )
        envs_dict[t_name] = e
        obs_dim, act_dim, low, high = o, a, l, h

    mt_env = MultiTaskEnv(
        envs_dict, max_episode_steps=cfg["defaults"]["max_episode_steps"]
    )

    agent = MTSACAgent(obs_dim, act_dim, len(tasks), low, high, device, sac_config)
    rb = ReplayBuffer(
        obs_dim + len(tasks), act_dim, cfg["multi_task"]["replay_buffer_size"]
    )

    print(f"\n--- [Multi-Task] {tasks} | Seed {seed} ---")
    print(
        f"    Params: Alpha={sac_config.init_alpha}, "
        f"Target Entropy Scale={sac_config.target_entropy_scale}, "
        f"Patience={patience}, Target={target_success}"
    )

    stats = run_training_loop(
        mt_env,
        agent,
        rb,
        cfg["multi_task"]["total_steps"],
        cfg["multi_task"]["start_steps"],
        cfg["multi_task"]["batch_size"],
        cfg["multi_task"]["eval_interval"],
        save_path,
        seed,
        task_name_for_log,
        target_success,
        patience,
        algo,
        is_multitask=True,
    )

    json_path = os.path.join(save_path, "results.json")
    full_log = {"config": cfg, "final_sac_params": raw_params, "results": stats}
    with open(json_path, "w") as f:
        json.dump(full_log, f, indent=4)
    print(f"Stats saved to {json_path}")

    mt_env.close()
