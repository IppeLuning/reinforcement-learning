from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import optax
from flax import struct
from flax.core import FrozenDict

from src.networks.actor import GaussianActor
from src.networks.critic import TwinQNetwork
from src.utils.types import Mask, Params, PRNGKey


class SACTrainState(struct.PyTreeNode):
    """Training state for Soft Actor-Critic with Lottery Ticket Mask support.

    This dataclass holds all mutable state required for SAC training. It is
    registered as a JAX PyTree, allowing it to be passed through `jax.jit`
    and `jax.grad`. It includes specialized logic to handle binary masks,
    ensuring that pruned parameters remain at zero after optimization steps.

    Attributes:
        step: Current training step counter.
        actor_params: Current parameters of the actor network.
        actor_opt_state: Optimizer state for the actor.
        critic_params: Current parameters of the twin Q-networks.
        critic_opt_state: Optimizer state for the critics.
        target_critic_params: Parameters of the target Q-networks (slow-moving).
        log_alpha: Learnable log-temperature parameter for entropy tuning.
        alpha_opt_state: Optimizer state for the alpha parameter.
        actor_apply_fn: The `apply` method of the actor Flax module.
        critic_apply_fn: The `apply` method of the critic Flax module.
        actor_optimizer: Optax transformation for the actor.
        critic_optimizer: Optax transformation for the critic.
        alpha_optimizer: Optax transformation for alpha.
        actor_mask: Optional binary mask for actor kernels.
        critic_mask: Optional binary mask for critic kernels.
    """

    step: int

    # Actor state
    actor_params: Params
    actor_opt_state: optax.OptState

    # Critic state
    critic_params: Params
    critic_opt_state: optax.OptState
    target_critic_params: Params

    # Entropy Temperature state
    log_alpha: jax.Array
    alpha_opt_state: optax.OptState

    # Static metadata (not PyTree nodes)
    actor_apply_fn: Callable = struct.field(pytree_node=False)
    critic_apply_fn: Callable = struct.field(pytree_node=False)
    actor_optimizer: optax.GradientTransformation = struct.field(pytree_node=False)
    critic_optimizer: optax.GradientTransformation = struct.field(pytree_node=False)
    alpha_optimizer: optax.GradientTransformation = struct.field(pytree_node=False)

    # Pruning Masks
    actor_mask: Optional[Mask] = None
    critic_mask: Optional[Mask] = None

    @property
    def alpha(self) -> jax.Array:
        """Computes the current temperature value $\alpha$."""
        return jnp.exp(self.log_alpha)

    def apply_actor_update(self, grads: Params) -> SACTrainState:
        """Updates actor parameters and enforces sparsity masks.

        Args:
            grads: Gradients of the actor loss with respect to actor_params.

        Returns:
            A new SACTrainState with updated actor parameters.
        """
        updates, new_opt_state = self.actor_optimizer.update(
            grads, self.actor_opt_state, self.actor_params
        )
        new_params = optax.apply_updates(self.actor_params, updates)

        if self.actor_mask is not None:
            new_params = jax.tree.map(lambda p, m: p * m, new_params, self.actor_mask)

        return self.replace(
            actor_params=new_params,
            actor_opt_state=new_opt_state,
        )

    def apply_critic_update(self, grads: Params) -> SACTrainState:
        """Updates critic parameters and enforces sparsity masks.

        Args:
            grads: Gradients of the critic loss with respect to critic_params.

        Returns:
            A new SACTrainState with updated critic parameters.
        """
        updates, new_opt_state = self.critic_optimizer.update(
            grads, self.critic_opt_state, self.critic_params
        )
        new_params = optax.apply_updates(self.critic_params, updates)

        if self.critic_mask is not None:
            new_params = jax.tree.map(lambda p, m: p * m, new_params, self.critic_mask)

        return self.replace(
            critic_params=new_params,
            critic_opt_state=new_opt_state,
        )

    def apply_alpha_update(self, grads: jax.Array) -> SACTrainState:
        """Updates the entropy temperature parameter.

        Args:
            grads: Gradients of the alpha loss with respect to log_alpha.

        Returns:
            A new SACTrainState with updated log_alpha.
        """
        updates, new_opt_state = self.alpha_optimizer.update(
            grads, self.alpha_opt_state
        )
        new_log_alpha = optax.apply_updates(self.log_alpha, updates)
        return self.replace(
            log_alpha=new_log_alpha,
            alpha_opt_state=new_opt_state,
        )

    def soft_update_target(self, tau: float) -> SACTrainState:
        """Updates target critic parameters using Polyak averaging.

        $$ \theta_{target} \leftarrow \tau \theta + (1 - \tau) \theta_{target} $$

        Args:
            tau: Smoothing coefficient (typically close to 0).

        Returns:
            A new SACTrainState with updated target_critic_params.
        """
        new_target_params = jax.tree.map(
            lambda p, tp: tau * p + (1.0 - tau) * tp,
            self.critic_params,
            self.target_critic_params,
        )

        if self.critic_mask is not None:
            new_target_params = jax.tree.map(
                lambda p, m: p * m, new_target_params, self.critic_mask
            )

        return self.replace(target_critic_params=new_target_params)

    def increment_step(self) -> SACTrainState:
        """Increments the internal training step counter.

        Returns:
            A new SACTrainState with step + 1.
        """
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
    use_masking: bool = False,
) -> SACTrainState:
    """Initializes a new SACTrainState.

    Args:
        key: PRNGKey for parameter initialization.
        obs_dim: Observation space dimension.
        act_dim: Action space dimension.
        hidden_dims: Hidden layer sequence for both networks.
        actor_lr: Learning rate for the actor optimizer.
        critic_lr: Learning rate for the critic optimizer.
        alpha_lr: Learning rate for the temperature optimizer.
        init_alpha: Starting value for the temperature parameter.
        use_masking: Legacy flag for masking initialization.

    Returns:
        An initialized SACTrainState.
    """
    key, actor_key, critic_key = jax.random.split(key, 3)

    dummy_obs = jnp.ones((1, obs_dim))
    dummy_action = jnp.ones((1, act_dim))

    actor = GaussianActor(act_dim=act_dim, hidden_dims=hidden_dims)
    actor_params = actor.init(actor_key, dummy_obs)
    actor_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=actor_lr, weight_decay=1e-4),
    )
    actor_opt_state = actor_optimizer.init(actor_params)

    critic = TwinQNetwork(hidden_dims=hidden_dims)
    critic_params = critic.init(critic_key, dummy_obs, dummy_action)
    critic_optimizer = optax.adam(critic_lr)
    critic_opt_state = critic_optimizer.init(critic_params)

    target_critic_params = critic_params

    log_alpha = jnp.log(init_alpha)
    alpha_optimizer = optax.adam(alpha_lr)
    alpha_opt_state = alpha_optimizer.init(log_alpha)

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
