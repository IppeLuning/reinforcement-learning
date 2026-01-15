"""Soft Actor-Critic (SAC) agent implementation in JAX.

DISCLAIMER: This code was written by Claude Opus 4.5 on 2026-01-12
and reviewed by Marinus van den Ende.

This module provides a complete SAC implementation with:
- JIT-compiled training step for maximum performance
- Learnable temperature (alpha) with automatic entropy tuning
- Support for masked training (Lottery Ticket experiments) via 'use_masking' flag
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict

from src.data.replay_buffer import ReplayBuffer
from src.networks.actor import (
    GaussianActor,
    deterministic_action,
    sample_action,
    scale_action,
)
from src.networks.critic import TwinQNetwork
from src.training.train_state import (
    MaskedTrainState,
    SACTrainState,
    create_sac_train_state,
)
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
        target_entropy_scale: Scale factor for target entropy
            (target = -scale * act_dim).
        auto_alpha: Whether to automatically tune temperature.
        init_alpha: Initial temperature value.
        hidden_dims: Hidden layer dimensions for all networks.
        alpha_min: Minimum allowed alpha value (for stability).
        alpha_max: Maximum allowed alpha value (for stability).
        max_grad_norm: Maximum gradient norm for clipping (None to disable).
        use_masking: Whether to enforce binary masks during training (for Lottery Tickets).
    """

    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    target_entropy_scale: float = 1.0
    auto_alpha: bool = True
    init_alpha: float = 0.2  # Changed to 0.2 (standard safe default)
    hidden_dims: Tuple[int, ...] = (256, 256)
    alpha_min: float = 1e-4
    alpha_max: float = 10.0
    max_grad_norm: Optional[float] = 1.0
    use_masking: bool = False


class SACAgent:
    """Soft Actor-Critic agent with JIT-compiled training.

    This class provides a high-level interface for SAC training and inference.
    The core training logic is JIT-compiled for maximum performance on
    accelerators (GPU/TPU/Metal).

    Attributes:
        config: SACConfig with hyperparameters.
        obs_dim: Dimension of observation space.
        act_dim: Dimension of action space.
        action_bounds: ActionBounds with low/high limits.
        state: SACTrainState containing all learnable parameters.
        normalizer: RunningNormalizer for observation normalization.
        target_entropy: Target entropy for automatic temperature tuning.
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
        """Initialize SAC agent.

        Args:
            obs_dim: Dimension of observation space.
            act_dim: Dimension of action space.
            act_low: Lower bounds of action space.
            act_high: Upper bounds of action space.
            config: SACConfig with hyperparameters.
            seed: Random seed for initialization.
        """
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
        # Note: 'use_masking' is passed as a static argument
        self._train_step = jax.jit(
            partial(
                _sac_train_step,
                gamma=config.gamma,
                tau=config.tau,
                target_entropy=self.target_entropy,
                auto_alpha=config.auto_alpha,
                alpha_min=config.alpha_min,
                alpha_max=config.alpha_max,
                use_masking=config.use_masking,
            )
        )

    def select_action(
        self,
        obs: np.ndarray,
        eval_mode: bool = False,
    ) -> np.ndarray:
        """Select action given observation.

        Args:
            obs: Observation array of shape (obs_dim,).
            eval_mode: If True, use deterministic policy (no sampling).

        Returns:
            Action array of shape (act_dim,).
        """
        # Convert to JAX array and add batch dimension
        obs_jax = jnp.asarray(obs, dtype=jnp.float32)[None, :]

        # Normalize observation
        obs_norm = self.normalizer.normalize(obs_jax)

        # Get policy distribution
        mean, log_std = self.state.actor_apply_fn(self.state.actor_params, obs_norm)

        if eval_mode:
            # Deterministic action (evaluation)
            action = deterministic_action(mean)
        else:
            # Stochastic action (exploration)
            self.key, action_key = jax.random.split(self.key)
            action, _ = sample_action(mean, log_std, action_key)

        # Scale to action bounds
        action = scale_action(action, self.action_bounds.low, self.action_bounds.high)

        # Remove batch dimension and convert to numpy
        return np.asarray(action[0])

    def update(
        self,
        replay_buffer: ReplayBuffer,
        batch_size: int = 256,
    ) -> Metrics:
        """Perform one gradient update step.

        Args:
            replay_buffer: Buffer to sample transitions from.
            batch_size: Number of transitions per batch.

        Returns:
            Dictionary of training metrics.
        """
        # Sample batch
        self.key, sample_key, update_key = jax.random.split(self.key, 3)
        batch = replay_buffer.sample(batch_size, key=sample_key)

        # Update observation normalizer
        self.normalizer = self.normalizer.update(batch.obs)
        self.normalizer = self.normalizer.update(batch.next_obs)

        # Normalize observations in batch
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
        """Current temperature value."""
        return float(self.state.alpha)

    def get_params(self) -> Tuple[Params, Params]:
        """Get current actor and critic parameters.

        Returns:
            Tuple of (actor_params, critic_params).
        """
        return self.state.actor_params, self.state.critic_params

    def set_masks(
        self,
        actor_mask: FrozenDict,
        critic_mask: FrozenDict,
    ) -> None:
        """Set masks for Lottery Ticket experiments.

        Converts the state to MaskedTrainState and applies masks.

        Args:
            actor_mask: Binary mask for actor parameters.
            critic_mask: Binary mask for critic parameters.
        """
        # Convert to MaskedTrainState
        self.state = MaskedTrainState(
            step=self.state.step,
            actor_params=self.state.actor_params,
            actor_opt_state=self.state.actor_opt_state,
            critic_params=self.state.critic_params,
            critic_opt_state=self.state.critic_opt_state,
            target_critic_params=self.state.target_critic_params,
            log_alpha=self.state.log_alpha,
            alpha_opt_state=self.state.alpha_opt_state,
            actor_apply_fn=self.state.actor_apply_fn,
            critic_apply_fn=self.state.critic_apply_fn,
            actor_optimizer=self.state.actor_optimizer,
            critic_optimizer=self.state.critic_optimizer,
            alpha_optimizer=self.state.alpha_optimizer,
            actor_mask=actor_mask,
            critic_mask=critic_mask,
        )


def _sac_train_step(
    state: Union[SACTrainState, MaskedTrainState],
    batch: Batch,
    key: PRNGKey,
    gamma: float,
    tau: float,
    target_entropy: float,
    auto_alpha: bool,
    alpha_min: float,
    alpha_max: float,
    use_masking: bool,
) -> Tuple[SACTrainState, Metrics]:
    """Single SAC training step (JIT-compiled).

    Performs one complete update:
    1. Critic update (minimize TD error)
    2. Actor update (maximize Q - alpha * log_prob)
    3. Alpha update (if auto_alpha, tune entropy)
    4. Target network soft update
    5. Mask enforcement (if use_masking is True)

    Args:
        state: Current training state.
        batch: Batch of transitions.
        key: Random key for action sampling.
        gamma: Discount factor.
        tau: Soft update coefficient.
        target_entropy: Target entropy for alpha tuning.
        auto_alpha: Whether to automatically tune alpha.
        alpha_min: Minimum alpha value.
        alpha_max: Maximum alpha value.
        use_masking: Whether to apply masking at end of step.

    Returns:
        Tuple of (updated state, metrics dict).
    """
    key, actor_key, target_key = jax.random.split(key, 3)

    # Current alpha value
    alpha = state.alpha

    # =========================================================================
    # Critic Update
    # =========================================================================

    def critic_loss_fn(critic_params: Params) -> Tuple[jax.Array, Metrics]:
        """Compute critic loss (TD error)."""
        # Current Q-values
        q1, q2 = state.critic_apply_fn(critic_params, batch.obs, batch.actions)

        # Target Q-values
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

        # MSE loss for both critics
        q1_loss = jnp.mean((q1 - target_q) ** 2)
        q2_loss = jnp.mean((q2 - target_q) ** 2)
        critic_loss = q1_loss + q2_loss

        metrics = {
            "critic_loss": critic_loss,
            "q1_mean": jnp.mean(q1),
            "q2_mean": jnp.mean(q2),
        }

        return critic_loss, metrics

    critic_grads, critic_metrics = jax.grad(critic_loss_fn, has_aux=True)(
        state.critic_params
    )
    state = state.apply_critic_update(critic_grads)

    # =========================================================================
    # Actor Update
    # =========================================================================

    def actor_loss_fn(actor_params: Params) -> Tuple[jax.Array, Metrics]:
        """Compute actor loss (maximize Q - alpha * entropy)."""
        mean, log_std = state.actor_apply_fn(actor_params, batch.obs)
        action, log_prob = sample_action(mean, log_std, actor_key)

        q1, q2 = state.critic_apply_fn(state.critic_params, batch.obs, action)
        q_min = jnp.minimum(q1, q2)

        # Actor tries to maximize Q-value while maintaining entropy
        actor_loss = jnp.mean(alpha * log_prob - q_min)

        metrics = {
            "actor_loss": actor_loss,
            "entropy": -jnp.mean(log_prob),
        }

        return actor_loss, (log_prob, metrics)

    actor_grads, (log_prob, actor_metrics) = jax.grad(actor_loss_fn, has_aux=True)(
        state.actor_params
    )
    state = state.apply_actor_update(actor_grads)

    # =========================================================================
    # Alpha Update (if enabled)
    # =========================================================================

    def alpha_loss_fn(log_alpha: jax.Array) -> jax.Array:
        """Compute alpha loss for entropy tuning."""
        return -jnp.mean(
            jnp.exp(log_alpha) * (jax.lax.stop_gradient(log_prob) + target_entropy)
        )

    if auto_alpha:
        alpha_grads = jax.grad(alpha_loss_fn)(state.log_alpha)
        state = state.apply_alpha_update(alpha_grads)

        # Clamp alpha to valid range
        new_log_alpha = jnp.clip(
            state.log_alpha,
            jnp.log(alpha_min),
            jnp.log(alpha_max),
        )
        state = state.replace(log_alpha=new_log_alpha)

    # =========================================================================
    # Target Network Soft Update
    # =========================================================================

    state = state.soft_update_target(tau)
    state = state.increment_step()

    # =========================================================================
    # Mask Enforcement (Lottery Ticket Hypothesis)
    # =========================================================================

    if use_masking:
        # NOTE: State must be MaskedTrainState if use_masking is True.
        # This forces pruned weights back to zero after the gradient update.
        state = state.apply_masks()

    # Combine all metrics
    metrics = {
        **critic_metrics,
        **actor_metrics,
        "alpha": state.alpha,
        "step": state.step,
    }

    return state, metrics
