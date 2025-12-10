import os
import random

# Import your actor classes from the training file
# Adjust "test" to the actual name of your training module.
from test import SACActor, TD3Actor, set_seed

import gymnasium as gym
import metaworld
import numpy as np
import torch


def make_metaworld_env_render(task_name: str, seed: int = 0):
    """
    Create a MetaWorld env with a visible Mujoco window (render_mode='human').
    task_name: 'reach-v3', 'push-v3', 'pick-place-v3'
    """
    assert task_name in ["reach-v3", "push-v3", "pick-place-v3"]

    ml1 = metaworld.ML1(task_name)
    env_cls = ml1.train_classes[task_name]

    # Ask for a human window
    env = env_cls(render_mode="human")
    task = random.choice(ml1.train_tasks)
    env.set_task(task)

    env = gym.wrappers.TimeLimit(env, max_episode_steps=200)

    obs, info = env.reset(seed=seed)
    return env


def load_actor(algo: str, task_name: str, device: str = "cpu"):
    """
    Rebuild and load the best actor for (algo, task_name) from checkpoints.
    """
    # Temporary env to get obs_dim, act_dim, bounds
    tmp_env = make_metaworld_env_render(task_name, seed=0)
    obs_dim = tmp_env.observation_space.shape[0]
    act_dim = tmp_env.action_space.shape[0]
    act_low = tmp_env.action_space.low
    act_high = tmp_env.action_space.high
    tmp_env.close()

    # Build actor
    algo = algo.lower()
    if algo == "sac":
        actor = SACActor(obs_dim, act_dim, act_low, act_high).to(device)
    elif algo == "td3":
        from test import TD3Actor  # import here if not already imported

        actor = TD3Actor(obs_dim, act_dim, act_low, act_high).to(device)
    else:
        raise ValueError("algo must be 'sac' or 'td3'")

    # Load weights
    ckpt_path = os.path.join("checkpoints", f"{algo}_{task_name}_best_actor.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device)
    actor.load_state_dict(state_dict)
    actor.eval()

    return actor


def watch_policy(
    algo: str = "sac",
    task_name: str = "reach-v3",
    episodes: int = 5,
    seed: int = 0,
):
    """
    Open a Mujoco window and run the trained policy for a few episodes.
    """
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = make_metaworld_env_render(task_name, seed=seed)
    actor = load_actor(algo, task_name, device=device)

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
                if algo.lower() == "sac":
                    # deterministic mode for SAC
                    action = actor.deterministic(obs_t).cpu().numpy()[0]
                else:  # td3
                    action = actor(obs_t).cpu().numpy()[0]

            obs, rew, done, truncated, info = env.step(action)
            ep_ret += rew

            # Render the frame (Mujoco window)
            env.render()

        print(f"[WATCH] Episode {ep} return: {ep_ret:.2f}")

    env.close()


if __name__ == "__main__":
    # Example: watch SAC policy on reach-v3
    watch_policy(
        algo="sac",
        task_name="reach-v3",
        episodes=3,
        seed=0,
    )
