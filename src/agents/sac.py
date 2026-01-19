"""Soft Actor-Critic (SAC) agent implementation in JAX.

DISCLAIMER: This code was written by Claude Opus 4.5 on 2026-01-12
and reviewed by Marinus van den Ende.

This module provides a complete SAC implementation with:
- JIT-compiled training step for maximum performance
- Learnable temperature (alpha) with automatic entropy tuning
- Intrinsic support for masked training (Lottery Ticket experiments)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np

from src.data.replay_buffer import ReplayBuffer
from src.networks.actor import deterministic_action, sample_action, scale_action
from src.training.train_state import SACTrainState, create_sac_train_state
from src.utils.normalizer import RunningNormalizer
from src.utils.types import ActionBounds, Batch, Metrics, Params, PRNGKey


@dataclass
class SACConfig:
    """Configuration for SAC agent.

    Attributes:
        gamma: Discount factor for future rewards.
        tau: Soft update coefficient for target networks.
        actor_lr: Learning rate for actor network.
        critic_lr: Learning rate for critic networks.
        alpha_lr: Learning rate for temperature parameter.
        target_entropy_scale: Scale factor for target entropy.
        auto_alpha: Whether to automatically tune temperature.
        init_alpha: Initial temperature value.
        hidden_dims: Hidden layer dimensions for all networks.
        alpha_min: Minimum allowed alpha value.
        alpha_max: Maximum allowed alpha value.
        max_grad_norm: Maximum gradient norm for clipping.
        use_masking: (Legacy/Flag) Hints if masking is expected, though
                     actual masking depends on state.actor_mask being set.
    """

    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    target_entropy_scale: float = 1.0
    auto_alpha: bool = True
    init_alpha: float = 0.2
    hidden_dims: Tuple[int, ...] = (256, 256)
    alpha_min: float = 1e-4
    alpha_max: float = 10.0
    max_grad_norm: Optional[float] = 1.0
    use_masking: bool = False


class SACAgent:
    """Soft Actor-Critic agent with JIT-compiled training.

    Attributes:
        config: SACConfig with hyperparameters.
        state: SACTrainState containing all learnable parameters & masks.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_low: np.ndarray,
        act_high: np.ndarray,
        config: SACConfig,
        seed: int = 0,
    ):
        """Initialize SAC agent."""
        self.config = config
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.action_bounds = ActionBounds.from_numpy(act_low, act_high)

        # Target entropy for automatic alpha tuning
        self.target_entropy = -config.target_entropy_scale * act_dim

        # Initialize random key
        self.key = jax.random.PRNGKey(seed)
        self.key, init_key = jax.random.split(self.key)

        # Create training state
        self.state = create_sac_train_state(
            key=init_key,
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dims=config.hidden_dims,
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
            alpha_lr=config.alpha_lr,
            init_alpha=config.init_alpha,
        )

        # Observation normalizer
        self.normalizer = RunningNormalizer.create(obs_dim)

        # JIT compile the training step
        self._train_step = jax.jit(
            partial(
                _sac_train_step,
                gamma=config.gamma,
                tau=config.tau,
                target_entropy=self.target_entropy,
                auto_alpha=config.auto_alpha,
                alpha_min=config.alpha_min,
                alpha_max=config.alpha_max,
                # Note: 'use_masking' logic is now intrinsic to SACTrainState
            )
        )

    def select_action(
        self,
        obs: np.ndarray,
        eval_mode: bool = False,
    ) -> np.ndarray:
        """Select action given observation."""
        obs_jax = jnp.asarray(obs, dtype=jnp.float32)[None, :]
        obs_norm = self.normalizer.normalize(obs_jax)

        mean, log_std = self.state.actor_apply_fn(self.state.actor_params, obs_norm)

        if eval_mode:
            action = deterministic_action(mean)
        else:
            self.key, action_key = jax.random.split(self.key)
            action, _ = sample_action(mean, log_std, action_key)

        action = scale_action(action, self.action_bounds.low, self.action_bounds.high)
        return np.asarray(action[0])

    def update(
        self,
        replay_buffer: ReplayBuffer,
        batch_size: int = 256,
    ) -> Metrics:
        """Perform one gradient update step."""
        self.key, sample_key, update_key = jax.random.split(self.key, 3)
        batch = replay_buffer.sample(batch_size, key=sample_key)

        # Update and apply normalizer
        self.normalizer = self.normalizer.update(batch.obs)
        self.normalizer = self.normalizer.update(batch.next_obs)

        batch = Batch(
            obs=self.normalizer.normalize(batch.obs),
            actions=batch.actions,
            rewards=batch.rewards,
            next_obs=self.normalizer.normalize(batch.next_obs),
            dones=batch.dones,
        )

        # Perform training step
        self.state, metrics = self._train_step(
            self.state,
            batch,
            update_key,
        )

        return metrics

    @property
    def alpha(self) -> float:
        return float(self.state.alpha)

    def get_params(self) -> Tuple[Params, Params]:
        return self.state.actor_params, self.state.critic_params

    def apply_mask(self, actor_mask: Params, critic_mask: Params) -> None:
        """
        Injects pruning masks into the agent's state.
        This enables sparse training for LTH experiments.
        """
        if not hasattr(self.state, "actor_mask") or not hasattr(
            self.state, "critic_mask"
        ):
            # This safety check should pass now that SACTrainState has these fields
            raise AttributeError("SACAgent state is missing mask fields.")

        # In JAX/Flax, states are immutable. replace() creates a new object.
        self.state = self.state.replace(actor_mask=actor_mask, critic_mask=critic_mask)
        print(f"  [SACAgent] Pruning masks applied. Future updates will be sparse.")


def _sac_train_step(
    state: SACTrainState,
    batch: Batch,
    key: PRNGKey,
    gamma: float,
    tau: float,
    target_entropy: float,
    auto_alpha: bool,
    alpha_min: float,
    alpha_max: float,
) -> Tuple[SACTrainState, Metrics]:
    """Single SAC training step (JIT-compiled)."""

    key, actor_key, target_key = jax.random.split(key, 3)
    alpha = state.alpha

    # --- 1. Critic Update ---
    def critic_loss_fn(critic_params: Params) -> Tuple[jax.Array, Metrics]:
        q1, q2 = state.critic_apply_fn(critic_params, batch.obs, batch.actions)

        # Target Q calculation
        next_mean, next_log_std = state.actor_apply_fn(
            state.actor_params, batch.next_obs
        )
        next_action, next_log_prob = sample_action(next_mean, next_log_std, target_key)

        target_q1, target_q2 = state.critic_apply_fn(
            state.target_critic_params, batch.next_obs, next_action
        )
        target_q = jnp.minimum(target_q1, target_q2) - alpha * next_log_prob
        target_q = batch.rewards + gamma * (1.0 - batch.dones) * target_q
        target_q = jax.lax.stop_gradient(target_q)

        loss = jnp.mean((q1 - target_q) ** 2) + jnp.mean((q2 - target_q) ** 2)
        return loss, {"critic_loss": loss, "q1": jnp.mean(q1), "q2": jnp.mean(q2)}

    critic_grads, critic_metrics = jax.grad(critic_loss_fn, has_aux=True)(
        state.critic_params
    )
    state = state.apply_critic_update(critic_grads)

    # --- 2. Actor Update ---
    def actor_loss_fn(actor_params: Params) -> Tuple[jax.Array, Metrics]:
        mean, log_std = state.actor_apply_fn(actor_params, batch.obs)
        action, log_prob = sample_action(mean, log_std, actor_key)

        q1, q2 = state.critic_apply_fn(state.critic_params, batch.obs, action)
        q_min = jnp.minimum(q1, q2)

        loss = jnp.mean(alpha * log_prob - q_min)
        return loss, {"actor_loss": loss, "entropy": -jnp.mean(log_prob)}

    actor_grads, actor_metrics = jax.grad(actor_loss_fn, has_aux=True)(
        state.actor_params
    )
    state = state.apply_actor_update(actor_grads)

    # --- 3. Alpha Update ---
    if auto_alpha:

        def alpha_loss_fn(log_alpha: jax.Array) -> jax.Array:
            # [FIX] Re-compute log_prob cleanly instead of trying to unpack dict keys
            mean, log_std = state.actor_apply_fn(state.actor_params, batch.obs)
            _, log_prob = sample_action(mean, log_std, actor_key)

            # Alpha loss: E[-alpha * (log_pi + target_entropy)]
            return -jnp.mean(
                jnp.exp(log_alpha) * (jax.lax.stop_gradient(log_prob) + target_entropy)
            )

        alpha_grads = jax.grad(alpha_loss_fn)(state.log_alpha)
        state = state.apply_alpha_update(alpha_grads)

        # Clip alpha
        new_log_alpha = jnp.clip(
            state.log_alpha, jnp.log(alpha_min), jnp.log(alpha_max)
        )
        state = state.replace(log_alpha=new_log_alpha)

    # --- 4. Target Update ---
    state = state.soft_update_target(tau)
    state = state.increment_step()

    metrics = {
        **critic_metrics,
        **actor_metrics,
        "alpha": state.alpha,
        "step": state.step,
    }

    return state, metrics
