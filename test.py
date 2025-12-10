"""
sac_td3_metaworld.py

Minimal SAC and TD3 implementations for MetaWorld tasks:
 - reach-v3
 - push-v3
 - pick-place-v3

This is a research-friendly skeleton: clear structure, no logging framework,
and easy to plug into pruning / LTH code.
"""

import os
import random
from dataclasses import dataclass
from typing import Tuple

import gymnasium as gym
import metaworld
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Utils
# ============================================================


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=(256, 256), activation=nn.ReLU):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), activation()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size=int(1e6)):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size = size
        self.ptr = 0
        self.size = 0

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=256):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=torch.as_tensor(self.obs_buf[idxs], dtype=torch.float32),
            acts=torch.as_tensor(self.acts_buf[idxs], dtype=torch.float32),
            rews=torch.as_tensor(self.rews_buf[idxs], dtype=torch.float32).unsqueeze(
                -1
            ),
            next_obs=torch.as_tensor(self.next_obs_buf[idxs], dtype=torch.float32),
            done=torch.as_tensor(self.done_buf[idxs], dtype=torch.float32).unsqueeze(
                -1
            ),
        )
        return batch


# ============================================================
# SAC
# ============================================================


class SACActor(nn.Module):
    def __init__(
        self,
        obs_dim,
        act_dim,
        act_low,
        act_high,
        hidden_dims=(256, 256),
        log_std_bounds=(-20, 2),
    ):
        super().__init__()
        self.net = MLP(obs_dim, 2 * act_dim, hidden_dims)
        self.log_std_bounds = log_std_bounds
        self.register_buffer("act_low", torch.as_tensor(act_low, dtype=torch.float32))
        self.register_buffer("act_high", torch.as_tensor(act_high, dtype=torch.float32))

    def forward(self, obs) -> Tuple[torch.Tensor, torch.Tensor]:
        mu_log_std = self.net(obs)
        mu, log_std = torch.chunk(mu_log_std, 2, dim=-1)
        log_std = torch.tanh(log_std)
        low, high = self.log_std_bounds
        log_std = low + 0.5 * (high - low) * (log_std + 1)
        std = log_std.exp()
        return mu, std

    def sample(self, obs):
        mu, std = self(obs)
        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample()
        tanh_z = torch.tanh(z)
        # Scale to env bounds
        action = self.act_low + (tanh_z + 1.0) * 0.5 * (self.act_high - self.act_low)

        # Log prob with tanh correction
        log_prob = normal.log_prob(z) - torch.log(1 - tanh_z.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic(self, obs):
        mu, std = self(obs)
        z = mu
        tanh_z = torch.tanh(z)
        action = self.act_low + (tanh_z + 1.0) * 0.5 * (self.act_high - self.act_low)
        return action


class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dims=(256, 256)):
        super().__init__()
        self.net = MLP(obs_dim + act_dim, 1, hidden_dims)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)


@dataclass
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    target_entropy_scale: float = 0.5  # target_entropy = -act_dim * scale
    auto_alpha: bool = True
    init_alpha: float = 0.2


class SACAgent:
    def __init__(
        self,
        obs_dim,
        act_dim,
        act_low,
        act_high,
        device="cpu",
        config: SACConfig = SACConfig(),
    ):
        self.device = device
        self.config = config

        self.actor = SACActor(obs_dim, act_dim, act_low, act_high).to(device)
        self.q1 = QNetwork(obs_dim, act_dim).to(device)
        self.q2 = QNetwork(obs_dim, act_dim).to(device)
        self.q1_target = QNetwork(obs_dim, act_dim).to(device)
        self.q2_target = QNetwork(obs_dim, act_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=config.critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=config.critic_lr)

        if config.auto_alpha:
            self.log_alpha = torch.tensor(
                np.log(config.init_alpha), requires_grad=True, device=device
            )
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)
            self.target_entropy = -config.target_entropy_scale * act_dim
        else:
            self.log_alpha = torch.tensor(np.log(config.init_alpha), device=device)
            self.alpha_opt = None
            self.target_entropy = None

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs, eval_mode=False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(
            0
        )
        with torch.no_grad():
            if eval_mode:
                action = self.actor.deterministic(obs_t)
            else:
                action, _ = self.actor.sample(obs_t)
        return action.cpu().numpy()[0]

    def update(self, replay_buffer: ReplayBuffer, batch_size=256):
        batch = replay_buffer.sample_batch(batch_size)
        obs = batch["obs"].to(self.device)
        acts = batch["acts"].to(self.device)
        rews = batch["rews"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        done = batch["done"].to(self.device)

        # Critic update
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_obs)
            q1_next = self.q1_target(next_obs, next_actions)
            q2_next = self.q2_target(next_obs, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            target_q = rews + (1 - done) * self.config.gamma * q_next

        q1_pred = self.q1(obs, acts)
        q2_pred = self.q2(obs, acts)
        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)
        critic_loss = q1_loss + q2_loss

        self.q1_opt.zero_grad()
        self.q2_opt.zero_grad()
        critic_loss.backward()
        self.q1_opt.step()
        self.q2_opt.step()

        # Actor update
        pi_actions, log_pi = self.actor.sample(obs)
        q1_pi = self.q1(obs, pi_actions)
        q2_pi = self.q2(obs, pi_actions)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * log_pi - q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Alpha update (entropy temperature)
        if self.config.auto_alpha:
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

        # Soft update of targets
        with torch.no_grad():
            for param, target_param in zip(
                self.q1.parameters(), self.q1_target.parameters()
            ):
                target_param.data.mul_(1 - self.config.tau)
                target_param.data.add_(self.config.tau * param.data)
            for param, target_param in zip(
                self.q2.parameters(), self.q2_target.parameters()
            ):
                target_param.data.mul_(1 - self.config.tau)
                target_param.data.add_(self.config.tau * param.data)


# ============================================================
# TD3
# ============================================================


class TD3Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_low, act_high, hidden_dims=(256, 256)):
        super().__init__()
        self.net = MLP(obs_dim, act_dim, hidden_dims)
        self.register_buffer("act_low", torch.as_tensor(act_low, dtype=torch.float32))
        self.register_buffer("act_high", torch.as_tensor(act_high, dtype=torch.float32))

    def forward(self, obs):
        x = self.net(obs)
        x = torch.tanh(x)
        action = self.act_low + (x + 1.0) * 0.5 * (self.act_high - self.act_low)
        return action


@dataclass
class TD3Config:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_delay: int = 2  # delayed policy update


class TD3Agent:
    def __init__(
        self,
        obs_dim,
        act_dim,
        act_low,
        act_high,
        device="cpu",
        config: TD3Config = TD3Config(),
    ):
        self.device = device
        self.config = config

        self.actor = TD3Actor(obs_dim, act_dim, act_low, act_high).to(device)
        self.actor_target = TD3Actor(obs_dim, act_dim, act_low, act_high).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.q1 = QNetwork(obs_dim, act_dim).to(device)
        self.q2 = QNetwork(obs_dim, act_dim).to(device)
        self.q1_target = QNetwork(obs_dim, act_dim).to(device)
        self.q2_target = QNetwork(obs_dim, act_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=config.critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=config.critic_lr)

        self.train_step = 0

    def select_action(self, obs, eval_mode=False, expl_noise=0.1):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(
            0
        )
        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy()[0]
        if not eval_mode and expl_noise > 0.0:
            action = action + expl_noise * np.random.randn(*action.shape)
        return action

    def update(self, replay_buffer: ReplayBuffer, batch_size=256):
        self.train_step += 1
        batch = replay_buffer.sample_batch(batch_size)
        obs = batch["obs"].to(self.device)
        acts = batch["acts"].to(self.device)
        rews = batch["rews"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        done = batch["done"].to(self.device)

        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(acts) * self.config.policy_noise).clamp(
                -self.config.noise_clip, self.config.noise_clip
            )
            next_actions = self.actor_target(next_obs) + noise
            # Clip to valid range by re-passing through tanh scaling
            # (the actor_target already outputs in bounds, noise may push it out)
            # Simple clipping:
            # NOTE: this assumes symmetric low/high, otherwise use env bounds buffer
            # and clamp accordingly.
            # For MetaWorld actions, low=-1, high=1 typically.
            next_actions = torch.clamp(next_actions, -1.0, 1.0)

            q1_next = self.q1_target(next_obs, next_actions)
            q2_next = self.q2_target(next_obs, next_actions)
            q_next = torch.min(q1_next, q2_next)
            target_q = rews + (1 - done) * self.config.gamma * q_next

        # Critic update
        q1_pred = self.q1(obs, acts)
        q2_pred = self.q2(obs, acts)
        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)
        critic_loss = q1_loss + q2_loss

        self.q1_opt.zero_grad()
        self.q2_opt.zero_grad()
        critic_loss.backward()
        self.q1_opt.step()
        self.q2_opt.step()

        # Delayed actor and target update
        if self.train_step % self.config.policy_delay == 0:
            # Actor loss: maximize Q, so minimize -Q
            q1_pi = self.q1(obs, self.actor(obs))
            actor_loss = -q1_pi.mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # Soft update targets
            with torch.no_grad():
                for param, target_param in zip(
                    self.q1.parameters(), self.q1_target.parameters()
                ):
                    target_param.data.mul_(1 - self.config.tau)
                    target_param.data.add_(self.config.tau * param.data)
                for param, target_param in zip(
                    self.q2.parameters(), self.q2_target.parameters()
                ):
                    target_param.data.mul_(1 - self.config.tau)
                    target_param.data.add_(self.config.tau * param.data)
                for param, target_param in zip(
                    self.actor.parameters(), self.actor_target.parameters()
                ):
                    target_param.data.mul_(1 - self.config.tau)
                    target_param.data.add_(self.config.tau * param.data)


# ============================================================
# MetaWorld env helper
# ============================================================


def make_metaworld_env(task_name: str, seed: int = 0):
    """
    task_name: 'reach-v3', 'push-v3', 'pick-place-v3'
    Returns: env, obs_dim, act_dim, act_low, act_high
    """
    assert task_name in ["reach-v3", "push-v3", "pick-place-v3"]

    ml1 = metaworld.ML1(task_name)  # single-task benchmark
    env_cls = ml1.train_classes[task_name]
    env = env_cls()
    task = random.choice(ml1.train_tasks)
    env.set_task(task)

    # MetaWorld envs are Gym-like
    # We wrap into gymnasium-compatible API if needed
    env = gym.wrappers.TimeLimit(env, max_episode_steps=200)

    # NEW: seeding in Gymnasium style
    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_low = env.action_space.low
    act_high = env.action_space.high

    return env, obs_dim, act_dim, act_low, act_high


# ============================================================
# Training loop
# ============================================================


def train(
    algo: str = "sac",  # 'sac' or 'td3'
    task_name: str = "reach-v3",
    total_steps: int = 200_000,
    start_steps: int = 10_000,
    batch_size: int = 1024,
    eval_interval: int = 10_000,
    seed: int = 0,
    target_return: float | None = None,  # e.g. 250.0 or None
    patience: int = 3,  # early-stop patience in eval steps
):

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env, obs_dim, act_dim, act_low, act_high = make_metaworld_env(task_name, seed)
    replay_buffer = ReplayBuffer(obs_dim, act_dim, size=int(1e6))

    if algo.lower() == "sac":
        agent = SACAgent(obs_dim, act_dim, act_low, act_high, device=device)
    elif algo.lower() == "td3":
        agent = TD3Agent(obs_dim, act_dim, act_low, act_high, device=device)
    else:
        raise ValueError("algo must be 'sac' or 'td3'")

    obs, _ = env.reset()
    episode_return = 0.0
    episode_len = 0

    # tracking best model and early stopping
    best_return = -np.inf
    no_improve = 0
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    for t in range(1, total_steps + 1):
        # Exploration vs exploitation
        if t < start_steps:
            act = env.action_space.sample()
        else:
            if algo.lower() == "td3":
                act = agent.select_action(obs, eval_mode=False, expl_noise=0.1)
            else:
                act = agent.select_action(obs, eval_mode=False)

        next_obs, rew, done, truncated, info = env.step(act)
        terminal = done or truncated

        replay_buffer.store(obs, act, rew, next_obs, float(done))

        obs = next_obs
        episode_return += rew
        episode_len += 1

        if terminal:
            print(
                f"Step {t} | Episode Return: {episode_return:.2f} | Length: {episode_len}"
            )
            obs, _ = env.reset()
            episode_return = 0.0
            episode_len = 0

        # Update
        if t >= start_steps:
            agent.update(replay_buffer, batch_size)

        # Simple evaluation + model saving + optional early stopping
        if t % eval_interval == 0:
            eval_return = evaluate(env, agent, episodes=5, algo=algo)
            print(
                f"[Eval] Step {t} | Task {task_name} | Algo {algo.upper()} | Return: {eval_return:.2f}"
            )

            # save best actor
            if eval_return > best_return + 1e-3:
                best_return = eval_return
                no_improve = 0
                actor_path = os.path.join(save_dir, f"{algo}_{task_name}_best_actor.pt")
                torch.save(agent.actor.state_dict(), actor_path)
                print(f"Saved best actor to {actor_path} (return={best_return:.2f})")
            else:
                no_improve += 1

            # optional early stopping
            if (
                target_return is not None
                and best_return >= target_return
                and no_improve >= patience
            ):
                print(
                    f"Early stopping at step {t}: best_return={best_return:.2f}, "
                    f"target_return={target_return}, no_improve={no_improve}"
                )
                break

    env.close()


def evaluate(env, agent, episodes: int = 5, algo: str = "sac") -> float:
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_ret = 0.0
        while not (done or truncated):
            if algo.lower() == "td3":
                act = agent.select_action(obs, eval_mode=True, expl_noise=0.0)
            else:
                act = agent.select_action(obs, eval_mode=True)
            obs, rew, done, truncated, info = env.step(act)
            ep_ret += rew
        returns.append(ep_ret)
    return float(np.mean(returns))


if __name__ == "__main__":
    # Example usage:
    # 1) SAC on reach-v3
    # train(algo="sac", task_name="reach-v3")
    #
    # 2) TD3 on push-v3
    # train(algo="td3", task_name="push-v3")
    #
    # 3) SAC on pick-place-v3
    # train(algo="sac", task_name="pick-place-v3")

    train(algo="sac", task_name="reach-v3")
