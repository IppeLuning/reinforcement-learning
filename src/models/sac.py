import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.mlp import MLP


class SACActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
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

        action = self.act_low + (tanh_z + 1.0) * 0.5 * (self.act_high - self.act_low)

        log_prob = normal.log_prob(z) - torch.log(1 - tanh_z.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic(self, obs):
        mu, _ = self(obs)
        tanh_z = torch.tanh(mu)
        action = self.act_low + (tanh_z + 1.0) * 0.5 * (self.act_high - self.act_low)
        return action


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims=(256, 256)):
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
    target_entropy_scale: float = 1.0  # target_entropy = -scale * act_dim
    auto_alpha: bool = True
    init_alpha: float = 0.2
    hidden_dims: Tuple[int, ...] = (256, 256)


class SACAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_low,
        act_high,
        device: str,
        config: SACConfig,
    ):
        self.device = device
        self.config = config

        # Extract hidden_dims from config to pass to networks
        h_dims = config.hidden_dims

        self.actor = SACActor(
            obs_dim, act_dim, act_low, act_high, hidden_dims=h_dims
        ).to(device)

        self.q1 = QNetwork(obs_dim, act_dim, hidden_dims=h_dims).to(device)
        self.q2 = QNetwork(obs_dim, act_dim, hidden_dims=h_dims).to(device)

        self.q1_target = QNetwork(obs_dim, act_dim, hidden_dims=h_dims).to(device)
        self.q2_target = QNetwork(obs_dim, act_dim, hidden_dims=h_dims).to(device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.q1_opt = torch.optim.Adam(self.q1.parameters(), lr=config.critic_lr)
        self.q2_opt = torch.optim.Adam(self.q2.parameters(), lr=config.critic_lr)

        init_alpha = float(config.init_alpha)

        if config.auto_alpha:
            self.log_alpha = torch.tensor(
                math.log(init_alpha),
                dtype=torch.float32,
                device=device,
                requires_grad=True,
            )
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)
            self.target_entropy = -config.target_entropy_scale * act_dim
        else:
            self.log_alpha = torch.tensor(
                math.log(init_alpha),
                dtype=torch.float32,
                device=device,
                requires_grad=False,
            )
            self.alpha_opt = None
            self.target_entropy = None

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs, eval_mode: bool = False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(
            0
        )
        with torch.no_grad():
            if eval_mode:
                action = self.actor.deterministic(obs_t)
            else:
                action, _ = self.actor.sample(obs_t)
        return action.cpu().numpy()[0]

    def update(self, replay_buffer, batch_size: int = 256):
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

        # Alpha update
        if self.config.auto_alpha:
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

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
