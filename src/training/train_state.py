"""Custom TrainState for Soft Actor-Critic.

DISCLAIMER: This code was written by Claude Opus 4.5 on 2026-01-12
and reviewed by Marinus van den Ende.

This module defines the state containers that hold all training parameters,
optimizer states, and masks for the SAC algorithm. The state is designed
to be compatible with JAX transformations like jit and vmap.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.core import FrozenDict
from flax.training import train_state

from src.networks.actor import GaussianActor
from src.networks.critic import TwinQNetwork
from src.utils.types import Mask, Params, PRNGKey


class SACTrainState(struct.PyTreeNode):
    """Training state for Soft Actor-Critic with Lottery Ticket Mask support.

    This dataclass holds all the mutable state needed for SAC training:
    - Actor parameters and optimizer state
    - Twin critic parameters, optimizer state, and target parameters
    - Learnable temperature (alpha) and its optimizer state
    - Training step counter
    - [NEW] Optional binary masks for pruning

    All fields are registered as pytree nodes, making the entire state
    compatible with jax.jit and other JAX transformations.

    Attributes:
        step: Current training step.
        actor_params: Parameters of the actor network.
        actor_opt_state: Optimizer state for the actor.
        critic_params: Parameters of the twin Q-networks.
        critic_opt_state: Optimizer state for the critics.
        target_critic_params: Parameters of the target Q-networks.
        log_alpha: Log of the temperature parameter (learnable).
        alpha_opt_state: Optimizer state for alpha.
        actor_apply_fn: Function to apply actor network (not a pytree leaf).
        critic_apply_fn: Function to apply critic network (not a pytree leaf).
        actor_optimizer: Actor optimizer (not a pytree leaf).
        critic_optimizer: Critic optimizer (not a pytree leaf).
        alpha_optimizer: Alpha optimizer (not a pytree leaf).
        actor_mask: Optional binary mask for actor (1=keep, 0=prune).
        critic_mask: Optional binary mask for critic.
    """

    # Training step counter
    step: int

    # Actor
    actor_params: Params
    actor_opt_state: optax.OptState

    # Twin Critics
    critic_params: Params
    critic_opt_state: optax.OptState
    target_critic_params: Params

    # Temperature (alpha)
    log_alpha: jax.Array
    alpha_opt_state: optax.OptState

    # Apply functions (static, not part of pytree)
    actor_apply_fn: Callable = struct.field(pytree_node=False)
    critic_apply_fn: Callable = struct.field(pytree_node=False)

    # Optimizers (static, not part of pytree)
    actor_optimizer: optax.GradientTransformation = struct.field(pytree_node=False)
    critic_optimizer: optax.GradientTransformation = struct.field(pytree_node=False)
    alpha_optimizer: optax.GradientTransformation = struct.field(pytree_node=False)

    # [NEW] Optional Masks (part of pytree so they can be jitted)
    actor_mask: Optional[Mask] = None
    critic_mask: Optional[Mask] = None

    @property
    def alpha(self) -> jax.Array:
        """Current temperature value (exp of log_alpha)."""
        return jnp.exp(self.log_alpha)

    def apply_actor_update(
        self,
        grads: Params,
    ) -> "SACTrainState":
        """Apply gradient update to actor parameters."""
        updates, new_opt_state = self.actor_optimizer.update(
            grads, self.actor_opt_state, self.actor_params
        )
        new_params = optax.apply_updates(self.actor_params, updates)

        # [CRITICAL] Re-apply mask to ensure pruned weights stay zero
        if self.actor_mask is not None:
            new_params = jax.tree.map(lambda p, m: p * m, new_params, self.actor_mask)

        return self.replace(
            actor_params=new_params,
            actor_opt_state=new_opt_state,
        )

    def apply_critic_update(
        self,
        grads: Params,
    ) -> "SACTrainState":
        """Apply gradient update to critic parameters."""
        updates, new_opt_state = self.critic_optimizer.update(
            grads, self.critic_opt_state, self.critic_params
        )
        new_params = optax.apply_updates(self.critic_params, updates)

        # [CRITICAL] Re-apply mask to ensure pruned weights stay zero
        if self.critic_mask is not None:
            new_params = jax.tree.map(lambda p, m: p * m, new_params, self.critic_mask)

        return self.replace(
            critic_params=new_params,
            critic_opt_state=new_opt_state,
        )

    def apply_alpha_update(
        self,
        grads: jax.Array,
    ) -> "SACTrainState":
        """Apply gradient update to temperature parameter."""
        updates, new_opt_state = self.alpha_optimizer.update(
            grads, self.alpha_opt_state
        )
        new_log_alpha = optax.apply_updates(self.log_alpha, updates)
        return self.replace(
            log_alpha=new_log_alpha,
            alpha_opt_state=new_opt_state,
        )

    def soft_update_target(self, tau: float) -> "SACTrainState":
        """Perform soft update of target critic parameters."""
        new_target_params = jax.tree.map(
            lambda p, tp: tau * p + (1.0 - tau) * tp,
            self.critic_params,
            self.target_critic_params,
        )
        # Note: If critic_params are masked, target will naturally become masked over time.
        # But we can enforce it explicitly if we want to be safe.
        if self.critic_mask is not None:
            new_target_params = jax.tree.map(
                lambda p, m: p * m, new_target_params, self.critic_mask
            )

        return self.replace(target_critic_params=new_target_params)

    def increment_step(self) -> "SACTrainState":
        """Increment the training step counter."""
        return self.replace(step=self.step + 1)


def create_sac_train_state(
    key: PRNGKey,
    obs_dim: int,
    act_dim: int,
    hidden_dims: Sequence[int] = (256, 256),
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    alpha_lr: float = 3e-4,
    init_alpha: float = 1.0,
    use_masking: bool = False,  # Legacy argument support (optional)
) -> SACTrainState:
    """Create and initialize a SACTrainState."""

    # Split keys for different initializations
    key, actor_key, critic_key = jax.random.split(key, 3)

    # Create dummy inputs for initialization
    dummy_obs = jnp.ones((1, obs_dim))
    dummy_action = jnp.ones((1, act_dim))

    # Initialize actor
    actor = GaussianActor(act_dim=act_dim, hidden_dims=hidden_dims)
    actor_params = actor.init(actor_key, dummy_obs)
    actor_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=actor_lr, weight_decay=1e-4),
    )
    actor_opt_state = actor_optimizer.init(actor_params)

    # Initialize twin critics
    critic = TwinQNetwork(hidden_dims=hidden_dims)
    critic_params = critic.init(critic_key, dummy_obs, dummy_action)
    critic_optimizer = optax.adam(critic_lr)
    critic_opt_state = critic_optimizer.init(critic_params)

    # Target critic starts as a copy
    target_critic_params = critic_params

    # Initialize learnable temperature
    log_alpha = jnp.log(init_alpha)
    alpha_optimizer = optax.adam(alpha_lr)
    alpha_opt_state = alpha_optimizer.init(log_alpha)

    # Return standard state (Masks default to None)
    return SACTrainState(
        step=0,
        actor_params=actor_params,
        actor_opt_state=actor_opt_state,
        critic_params=critic_params,
        critic_opt_state=critic_opt_state,
        target_critic_params=target_critic_params,
        log_alpha=log_alpha,
        alpha_opt_state=alpha_opt_state,
        actor_apply_fn=actor.apply,
        critic_apply_fn=critic.apply,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        alpha_optimizer=alpha_optimizer,
        actor_mask=None,
        critic_mask=None,
    )
