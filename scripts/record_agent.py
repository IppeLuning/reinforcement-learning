import argparse
import os

import gymnasium as gym
import imageio
import metaworld
import numpy as np
import torch
from gymnasium.wrappers import TimeLimit

# Import your existing classes
from src.models.sac import SACAgent, SACConfig
from src.utils.device import get_device


# --- Reusing your wrapper ---
class MetaWorldTaskSampler(gym.Wrapper):
    def __init__(self, env, benchmark, mode="test"):
        super().__init__(env)
        self.benchmark = benchmark
        self.mode = mode
        self.task_list = (
            benchmark.train_tasks if mode == "train" else benchmark.test_tasks
        )

    def reset(self, seed=None, options=None):
        if seed is not None:
            import random

            random.seed(seed)
            np.random.seed(seed)

        # Randomly select a task for this episode
        import random

        new_task = random.choice(self.task_list)
        self.env.set_task(new_task)
        return self.env.reset(seed=seed, options=options)


def make_eval_env(task_name, max_episode_steps, seed=None, camera_name=None):
    print(f"Creating environment for: {task_name}")
    ml1 = metaworld.ML1(task_name)
    env_cls = ml1.train_classes[task_name]

    # 1. Initialize with render_mode for video
    env = env_cls(render_mode="rgb_array")

    # 2. Apply Camera Fix immediately if requested
    if camera_name:
        try:
            # We must access the underlying MuJoCo model to find the ID
            model = env.unwrapped.model
            # Look up the ID associated with the name (e.g., "topview" -> 2)
            cam_id = -1
            for i in range(model.ncam):
                name = model.camera(i).name
                if name == camera_name:
                    cam_id = i
                    break

            if cam_id != -1:
                print(
                    f"🎥 Camera '{camera_name}' found (ID: {cam_id}). Setting renderer..."
                )
                # Set BOTH id and name to be safe
                env.unwrapped.mujoco_renderer.camera_id = cam_id
                env.unwrapped.mujoco_renderer.camera_name = camera_name
            else:
                print(f"⚠️ Camera '{camera_name}' not found. Using default.")
        except AttributeError:
            print("⚠️ Could not access MuJoCo model to set camera. Ignoring.")

    # 3. Apply Wrappers
    env = MetaWorldTaskSampler(env, ml1, mode="test")
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    if seed is not None:
        env.reset(seed=seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_low = env.action_space.low
    act_high = env.action_space.high
    return env, obs_dim, act_dim, act_low, act_high


def run_inference_video(
    checkpoint_path, task_name, output_prefix="video", seed=123, camera_name=None
):
    device = get_device()
    print(f"Running inference on {device}")

    # 1. Setup Env with Camera
    max_steps = 500
    env, obs_dim, act_dim, low, high = make_eval_env(
        task_name, max_steps, seed, camera_name
    )

    # 2. Setup Agent
    sac_config = SACConfig(
        hidden_dims=(400, 400, 400), actor_lr=0, critic_lr=0, alpha_lr=0
    )
    agent = SACAgent(obs_dim, act_dim, low, high, device, sac_config)

    # 3. Load Checkpoint (Smart Loader)
    print(f"Loading checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "normalizer_mean" in checkpoint:
        print("✅ Found updated checkpoint format. Loading Actor + Normalizer stats.")
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.obs_normalizer.mean = checkpoint["normalizer_mean"].to(device)
        agent.obs_normalizer.var = checkpoint["normalizer_var"].to(device)
        agent.obs_normalizer.count = checkpoint["normalizer_count"]
    else:
        print("⚠️  WARNING: Detected old checkpoint format (Actor weights only).")
        agent.actor.load_state_dict(checkpoint)

    agent.actor.eval()

    # 4. Run Episodes and Save Individually
    num_episodes = 5

    for i in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_ret = 0
        success = False

        # Local buffer for this episode's frames
        episode_frames = []

        print(f"Recording Episode {i+1}/{num_episodes}...")

        while not (done or truncated):
            raw_frame = env.render()
            # The [::-1] slice reverses the first dimension (height), flipping it 180 degrees vertically
            fixed_frame = raw_frame[::-1]
            episode_frames.append(fixed_frame)

            action = agent.select_action(obs, eval_mode=True)
            obs, reward, done, truncated, info = env.step(action)
            ep_ret += reward

            if info.get("success", 0.0) >= 1.0:
                success = True

        status = "SUCCESS" if success else "FAIL"
        print(f"   Finished. Return: {ep_ret:.2f} | Status: {status}")

        # Save THIS episode to a file immediately
        filename = f"videos/{output_prefix}_ep{i+1}_{status}_{ep_ret}.mp4"
        print(f"   Saving {filename}...")
        imageio.mimsave(filename, episode_frames, fps=30)

    env.close()
    print("All episodes recorded.")


if __name__ == "__main__":

    run_inference_video(
        "checkpoints/single_task/reach-v3/seed_0/sac_reach-v3_seed0_best_actor.pt",
        "reach-v3",
        "reach_agent.mp4",
        seed=42,
        camera_name="corner2",
    )
