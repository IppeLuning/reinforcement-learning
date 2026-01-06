import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils  # for gradient clipping

from src.utils.mlp import MLP
from src.utils.normalizer import RunningNormalizer


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

        # Scale from [-1, 1] to [act_low, act_high]
        action = self.act_low + (tanh_z + 1.0) * 0.5 * (self.act_high - self.act_low)

        # Log-prob with tanh correction
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

    # More conservative defaults
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    alpha_lr: float = 5e-5

    # Less aggressive entropy target by default
    target_entropy_scale: float = 0.5  # target_entropy = -scale * act_dim

    auto_alpha: bool = True
    init_alpha: float = 0.2
    hidden_dims: Tuple[int, ...] = (256, 256)

    # stability hooks
    alpha_min: float = 1e-4
    alpha_max: float = 10.0
    max_grad_norm: float = 1.0


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

        # If you use the lazy normalizer version:
        self.obs_normalizer = RunningNormalizer(device=device)
        # If your normalizer takes obs_dim in __init__, use instead:
        # self.obs_normalizer = RunningNormalizer(obs_dim, device=device)

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
        """
        Returns a numpy array of shape (act_dim,), never None.
        """
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(
            0
        )

        with torch.no_grad():
            norm_obs = self.obs_normalizer.normalize(obs_t)
            if eval_mode:
                action_t = self.actor.deterministic(norm_obs)
            else:
                action_t, _ = self.actor.sample(norm_obs)

        action_np = action_t.detach().cpu().numpy()
        # Expect shape (1, act_dim)
        if action_np.ndim == 2 and action_np.shape[0] == 1:
            action_np = action_np[0]
        return action_np

    def update(self, replay_buffer, batch_size: int = 256):
        batch = replay_buffer.sample_batch(batch_size)
        obs = batch["obs"].to(self.device)
        acts = batch["acts"].to(self.device)
        done = batch["done"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        rews = batch["rews"].to(self.device)

        # Update normalizer from this batch (no grad)
        with torch.no_grad():
            self.obs_normalizer.update(obs)
            self.obs_normalizer.update(next_obs)

        # Normalize before passing to networks
        obs_n = self.obs_normalizer.normalize(obs)
        next_obs_n = self.obs_normalizer.normalize(next_obs)

        # Critic update (use normalized obs)
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_obs_n)
            q1_next = self.q1_target(next_obs_n, next_actions)
            q2_next = self.q2_target(next_obs_n, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            target_q = rews + (1 - done) * self.config.gamma * q_next

        q1_pred = self.q1(obs_n, acts)
        q2_pred = self.q2(obs_n, acts)
        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)
        critic_loss = q1_loss + q2_loss

        self.q1_opt.zero_grad()
        self.q2_opt.zero_grad()
        critic_loss.backward()

        # gradient clipping for critics
        if self.config.max_grad_norm is not None and self.config.max_grad_norm > 0:
            nn_utils.clip_grad_norm_(self.q1.parameters(), self.config.max_grad_norm)
            nn_utils.clip_grad_norm_(self.q2.parameters(), self.config.max_grad_norm)

        self.q1_opt.step()
        self.q2_opt.step()

        # Actor update (use normalized obs)
        pi_actions, log_pi = self.actor.sample(obs_n)
        q1_pi = self.q1(obs_n, pi_actions)
        q2_pi = self.q2(obs_n, pi_actions)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * log_pi - q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()

        # gradient clipping for actor
        if self.config.max_grad_norm is not None and self.config.max_grad_norm > 0:
            nn_utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)

        self.actor_opt.step()

        # Alpha update
        if self.config.auto_alpha:
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

            # clamp log_alpha to keep alpha in [alpha_min, alpha_max]
            with torch.no_grad():
                self.log_alpha.data.clamp_(
                    math.log(self.config.alpha_min), math.log(self.config.alpha_max)
                )

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
