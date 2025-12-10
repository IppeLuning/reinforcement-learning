import os
import random

# Adjust "test" if your training script has a different name
from test import SACActor, TD3Actor, set_seed

import gymnasium as gym
import metaworld
import numpy as np
import torch
from gymnasium.wrappers import RecordVideo

# ------------------------------------------------------------
# Env helpers
# ------------------------------------------------------------


def make_base_metaworld_env(
    task_name: str, seed: int = 0, render_mode: str = "rgb_array"
):
    """
    Create a single MetaWorld ML1 env for a given task.
    task_name: 'reach-v3', 'push-v3', 'pick-place-v3'
    render_mode='rgb_array' so we can record video (no GLFW window).
    """
    assert task_name in ["reach-v3", "push-v3", "pick-place-v3"]

    ml1 = metaworld.ML1(task_name)  # single-task benchmark
    env_cls = ml1.train_classes[task_name]
    env = env_cls(render_mode=render_mode)
    task = random.choice(ml1.train_tasks)
    env.set_task(task)

    env = gym.wrappers.TimeLimit(env, max_episode_steps=200)
    obs, info = env.reset(seed=seed)

    return env


def make_video_env(task_name: str, seed: int = 0, video_folder: str = "videos"):
    """
    Wrap the base env with RecordVideo so that every episode is recorded to disk.
    """
    os.makedirs(video_folder, exist_ok=True)
    env = make_base_metaworld_env(task_name, seed=seed, render_mode="rgb_array")

    # Record every episode
    env = RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda episode_id: True,
        name_prefix=f"{task_name}",
    )

    return env


# ------------------------------------------------------------
# Actor loading
# ------------------------------------------------------------


def load_actor(algo: str, task_name: str, device: str = "cpu"):
    """
    Rebuild the actor network and load weights from checkpoints/{algo}_{task_name}_best_actor.pt
    """
    # Use a temporary env to infer dimensions and bounds
    tmp_env = make_base_metaworld_env(task_name, seed=0, render_mode="rgb_array")
    obs_dim = tmp_env.observation_space.shape[0]
    act_dim = tmp_env.action_space.shape[0]
    act_low = tmp_env.action_space.low
    act_high = tmp_env.action_space.high
    tmp_env.close()

    algo_l = algo.lower()
    if algo_l == "sac":
        actor = SACActor(obs_dim, act_dim, act_low, act_high).to(device)
    elif algo_l == "td3":
        actor = TD3Actor(obs_dim, act_dim, act_low, act_high).to(device)
    else:
        raise ValueError("algo must be 'sac' or 'td3'")

    ckpt_path = os.path.join("checkpoints", f"{algo_l}_{task_name}_best_actor.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device)
    actor.load_state_dict(state_dict)
    actor.eval()
    return actor


# ------------------------------------------------------------
# Rollout & video recording
# ------------------------------------------------------------


def record_policy_video(
    algo: str = "sac",
    task_name: str = "reach-v3",
    episodes: int = 3,
    seed: int = 0,
    video_folder: str = "videos",
):
    """
    Run the trained policy for a few episodes and record them to video_folder as .mp4.
    """
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    actor = load_actor(algo, task_name, device=device)
    env = make_video_env(task_name, seed=seed, video_folder=video_folder)

    algo_l = algo.lower()

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_ret = 0.0

        while not (done or truncated):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(
                0
            )
            with torch.no_grad():
                if algo_l == "sac":
                    # deterministic policy for visualization
                    action = actor.deterministic(obs_t).cpu().numpy()[0]
                else:  # td3
                    action = actor(obs_t).cpu().numpy()[0]

            obs, rew, done, truncated, info = env.step(action)
            ep_ret += rew

        print(f"[VIDEO] Episode {ep} return: {ep_ret:.2f}")

    env.close()
    print(f"Videos written to folder: {video_folder}")


if __name__ == "__main__":
    # Example: SAC on reach-v3
    record_policy_video(
        algo="sac",
        task_name="reach-v3",
        episodes=3,
        seed=0,
        video_folder="videos",
    )
