"""Multi-Task SAC Agent with shared encoder.

DISCLAIMER: This code was written by Claude Opus 4.5 on 2026-01-12
and reviewed by Marinus van den Ende.

This module implements a multi-task SAC agent that shares a common
encoder across tasks while optionally having task-specific heads.
This is the architecture used in the Lottery Ticket research for
discovering universal multi-task subnetworks.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax import struct
from flax.core import FrozenDict

from src.jax.agents.sac import SACAgent, SACConfig
from src.jax.buffers.replay_buffer import ReplayBuffer
from src.jax.networks.actor import sample_action, deterministic_action, scale_action
from src.jax.networks.mlp import MLP
from src.jax.training.train_state import SACTrainState, create_sac_train_state
from src.jax.utils.normalizer import RunningNormalizer
from src.jax.utils.types import PRNGKey, Params, Batch, Metrics, ActionBounds


# =============================================================================
# Multi-Task Network Architectures
# =============================================================================


class SharedEncoderActor(nn.Module):
    """Multi-task actor with shared encoder and task-conditioned output.
    
    The architecture consists of:
    1. Shared encoder layers that process the base observation
    2. Task embedding that conditions the policy on task identity
    3. Output layers that produce action distribution
    
    Attributes:
        act_dim: Dimension of action space.
        num_tasks: Number of tasks (for task embedding).
        encoder_dims: Dimensions of shared encoder layers.
        head_dims: Dimensions of task-conditioned head.
        log_std_bounds: Bounds for log standard deviation.
    """
    
    act_dim: int
    num_tasks: int
    encoder_dims: Sequence[int] = (256, 256)
    head_dims: Sequence[int] = (256,)
    log_std_bounds: Tuple[float, float] = (-20.0, 2.0)
    
    @nn.compact
    def __call__(
        self,
        obs: jax.Array,
        task_id: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """Forward pass with task conditioning.
        
        Args:
            obs: Base observations (without task one-hot).
            task_id: One-hot task identifier.
            
        Returns:
            Tuple of (mean, log_std) for the action distribution.
        """
        # Shared encoder
        x = obs
        for dim in self.encoder_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        
        # Task embedding (learned from one-hot)
        task_embed = nn.Dense(self.encoder_dims[-1])(task_id)
        task_embed = nn.relu(task_embed)
        
        # Combine encoder output with task embedding
        x = x + task_embed  # Additive conditioning
        
        # Task-conditioned head
        for dim in self.head_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        
        # Output mean and log_std
        output = nn.Dense(2 * self.act_dim)(x)
        mean, log_std = jnp.split(output, 2, axis=-1)
        
        # Bound log_std
        log_std_min, log_std_max = self.log_std_bounds
        log_std = jnp.tanh(log_std)
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1.0)
        
        return mean, log_std


class SharedEncoderCritic(nn.Module):
    """Multi-task twin Q-networks with shared encoder.
    
    Attributes:
        num_tasks: Number of tasks.
        encoder_dims: Dimensions of shared encoder.
        head_dims: Dimensions of Q-value heads.
    """
    
    num_tasks: int
    encoder_dims: Sequence[int] = (256, 256)
    head_dims: Sequence[int] = (256,)
    
    @nn.compact
    def __call__(
        self,
        obs: jax.Array,
        action: jax.Array,
        task_id: jax.Array,
    ) -> Tuple[jax.Array, jax.Array]:
        """Forward pass for twin Q-networks.
        
        Args:
            obs: Base observations.
            action: Actions taken.
            task_id: One-hot task identifier.
            
        Returns:
            Tuple of (q1, q2) Q-values.
        """
        # Concatenate inputs
        x = jnp.concatenate([obs, action], axis=-1)
        
        # Shared encoder
        for dim in self.encoder_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        
        # Task embedding
        task_embed = nn.Dense(self.encoder_dims[-1])(task_id)
        task_embed = nn.relu(task_embed)
        
        # Combine with task
        x = x + task_embed
        
        # Q1 head
        q1 = x
        for dim in self.head_dims:
            q1 = nn.Dense(dim, name=f"q1_head_{dim}")(q1)
            q1 = nn.relu(q1)
        q1 = nn.Dense(1, name="q1_out")(q1)
        
        # Q2 head (independent)
        q2 = x
        for dim in self.head_dims:
            q2 = nn.Dense(dim, name=f"q2_head_{dim}")(q2)
            q2 = nn.relu(q2)
        q2 = nn.Dense(1, name="q2_out")(q2)
        
        return q1, q2


# =============================================================================
# Multi-Task Train State
# =============================================================================


class MTSACTrainState(struct.PyTreeNode):
    """Training state for Multi-Task SAC.
    
    Similar to SACTrainState but with multi-task networks.
    """
    
    step: int
    actor_params: Params
    actor_opt_state: optax.OptState
    critic_params: Params
    critic_opt_state: optax.OptState
    target_critic_params: Params
    log_alpha: jax.Array
    alpha_opt_state: optax.OptState
    
    # Network apply functions
    actor_apply_fn: callable = struct.field(pytree_node=False)
    critic_apply_fn: callable = struct.field(pytree_node=False)
    
    # Optimizers
    actor_optimizer: optax.GradientTransformation = struct.field(pytree_node=False)
    critic_optimizer: optax.GradientTransformation = struct.field(pytree_node=False)
    alpha_optimizer: optax.GradientTransformation = struct.field(pytree_node=False)
    
    @property
    def alpha(self) -> jax.Array:
        return jnp.exp(self.log_alpha)
    
    def soft_update_target(self, tau: float) -> "MTSACTrainState":
        new_target_params = jax.tree.map(
            lambda p, tp: tau * p + (1.0 - tau) * tp,
            self.critic_params,
            self.target_critic_params,
        )
        return self.replace(target_critic_params=new_target_params)


def create_mtsac_train_state(
    key: PRNGKey,
    obs_dim: int,
    act_dim: int,
    num_tasks: int,
    encoder_dims: Sequence[int] = (256, 256),
    head_dims: Sequence[int] = (256,),
    actor_lr: float = 3e-4,
    critic_lr: float = 3e-4,
    alpha_lr: float = 3e-4,
    init_alpha: float = 1.0,
) -> MTSACTrainState:
    """Create and initialize multi-task SAC training state.
    
    Args:
        key: Random key for initialization.
        obs_dim: Base observation dimension (without task one-hot).
        act_dim: Action dimension.
        num_tasks: Number of tasks.
        encoder_dims: Shared encoder layer dimensions.
        head_dims: Task-specific head dimensions.
        actor_lr: Actor learning rate.
        critic_lr: Critic learning rate.
        alpha_lr: Alpha learning rate.
        init_alpha: Initial temperature.
        
    Returns:
        Initialized MTSACTrainState.
    """
    key, actor_key, critic_key = jax.random.split(key, 3)
    
    # Dummy inputs
    dummy_obs = jnp.ones((1, obs_dim))
    dummy_action = jnp.ones((1, act_dim))
    dummy_task = jnp.ones((1, num_tasks))
    
    # Initialize actor
    actor = SharedEncoderActor(
        act_dim=act_dim,
        num_tasks=num_tasks,
        encoder_dims=encoder_dims,
        head_dims=head_dims,
    )
    actor_params = actor.init(actor_key, dummy_obs, dummy_task)
    actor_optimizer = optax.adam(actor_lr)
    actor_opt_state = actor_optimizer.init(actor_params)
    
    # Initialize critic
    critic = SharedEncoderCritic(
        num_tasks=num_tasks,
        encoder_dims=encoder_dims,
        head_dims=head_dims,
    )
    critic_params = critic.init(critic_key, dummy_obs, dummy_action, dummy_task)
    critic_optimizer = optax.adam(critic_lr)
    critic_opt_state = critic_optimizer.init(critic_params)
    
    # Target critic
    target_critic_params = critic_params
    
    # Alpha
    log_alpha = jnp.log(init_alpha)
    alpha_optimizer = optax.adam(alpha_lr)
    alpha_opt_state = alpha_optimizer.init(log_alpha)
    
    return MTSACTrainState(
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
    )


# =============================================================================
# Multi-Task SAC Agent
# =============================================================================


class MTSACAgent:
    """Multi-Task Soft Actor-Critic agent.
    
    Extends SACAgent for multi-task learning with:
    - Shared encoder across tasks
    - Task-conditioned policy and value functions
    - Support for task-augmented observations
    
    Attributes:
        config: SACConfig with hyperparameters.
        obs_dim: Base observation dimension (without task encoding).
        act_dim: Action dimension.
        num_tasks: Number of tasks.
        state: MTSACTrainState with parameters.
        
    Example:
        >>> agent = MTSACAgent(
        ...     obs_dim=10, act_dim=4, num_tasks=3,
        ...     act_low=-1.0, act_high=1.0,
        ...     config=SACConfig(), seed=42
        ... )
        >>> # Observation includes task one-hot
        >>> obs_with_task = np.concatenate([obs, task_one_hot])
        >>> action = agent.select_action(obs_with_task)
    """
    
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        num_tasks: int,
        act_low: np.ndarray,
        act_high: np.ndarray,
        config: SACConfig,
        seed: int = 0,
    ):
        """Initialize multi-task SAC agent.
        
        Args:
            obs_dim: Base observation dimension.
            act_dim: Action dimension.
            num_tasks: Number of tasks.
            act_low: Action lower bounds.
            act_high: Action upper bounds.
            config: SAC configuration.
            seed: Random seed.
        """
        self.config = config
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_tasks = num_tasks
        self.action_bounds = ActionBounds.from_numpy(act_low, act_high)
        
        # Target entropy
        self.target_entropy = -config.target_entropy_scale * act_dim
        
        # Random key
        self.key = jax.random.PRNGKey(seed)
        self.key, init_key = jax.random.split(self.key)
        
        # Create training state
        self.state = create_mtsac_train_state(
            key=init_key,
            obs_dim=obs_dim,
            act_dim=act_dim,
            num_tasks=num_tasks,
            encoder_dims=config.hidden_dims,
            head_dims=(config.hidden_dims[-1],) if config.hidden_dims else (256,),
            actor_lr=config.actor_lr,
            critic_lr=config.critic_lr,
            alpha_lr=config.alpha_lr,
            init_alpha=config.init_alpha,
        )
        
        # Normalizer for base observations
        self.normalizer = RunningNormalizer.create(obs_dim)
        
        # JIT compile training step
        self._train_step = jax.jit(partial(
            _mtsac_train_step,
            gamma=config.gamma,
            tau=config.tau,
            target_entropy=self.target_entropy,
            auto_alpha=config.auto_alpha,
            alpha_min=config.alpha_min,
            alpha_max=config.alpha_max,
            num_tasks=num_tasks,
            obs_dim=obs_dim,
        ))
    
    def _split_obs(self, obs: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Split combined observation into base obs and task one-hot.
        
        Args:
            obs: Combined observation of shape (..., obs_dim + num_tasks).
            
        Returns:
            Tuple of (base_obs, task_id).
        """
        base_obs = obs[..., :self.obs_dim]
        task_id = obs[..., self.obs_dim:]
        return base_obs, task_id
    
    def select_action(
        self,
        obs: np.ndarray,
        eval_mode: bool = False,
    ) -> np.ndarray:
        """Select action given task-augmented observation.
        
        Args:
            obs: Observation with task one-hot appended.
            eval_mode: If True, use deterministic policy.
            
        Returns:
            Action array.
        """
        obs_jax = jnp.asarray(obs, dtype=jnp.float32)[None, :]
        base_obs, task_id = self._split_obs(obs_jax)
        
        # Normalize base observation
        base_obs_norm = self.normalizer.normalize(base_obs)
        
        # Get policy distribution
        mean, log_std = self.state.actor_apply_fn(
            self.state.actor_params, base_obs_norm, task_id
        )
        
        if eval_mode:
            action = deterministic_action(mean)
        else:
            self.key, action_key = jax.random.split(self.key)
            action, _ = sample_action(mean, log_std, action_key)
        
        # Scale to action bounds
        action = scale_action(action, self.action_bounds.low, self.action_bounds.high)
        
        return np.asarray(action[0])
    
    def update(
        self,
        replay_buffer: ReplayBuffer,
        batch_size: int = 256,
    ) -> Metrics:
        """Perform one gradient update.
        
        Args:
            replay_buffer: Buffer with task-augmented observations.
            batch_size: Batch size.
            
        Returns:
            Training metrics.
        """
        self.key, sample_key, update_key = jax.random.split(self.key, 3)
        batch = replay_buffer.sample(batch_size, key=sample_key)
        
        # Update normalizer with base observations
        base_obs = batch.obs[:, :self.obs_dim]
        base_next_obs = batch.next_obs[:, :self.obs_dim]
        self.normalizer = self.normalizer.update(base_obs)
        self.normalizer = self.normalizer.update(base_next_obs)
        
        # Normalize base observations in batch
        norm_obs = jnp.concatenate([
            self.normalizer.normalize(base_obs),
            batch.obs[:, self.obs_dim:],
        ], axis=-1)
        norm_next_obs = jnp.concatenate([
            self.normalizer.normalize(base_next_obs),
            batch.next_obs[:, self.obs_dim:],
        ], axis=-1)
        
        batch = Batch(
            obs=norm_obs,
            actions=batch.actions,
            rewards=batch.rewards,
            next_obs=norm_next_obs,
            dones=batch.dones,
        )
        
        self.state, metrics = self._train_step(self.state, batch, update_key)
        return metrics
    
    @property
    def alpha(self) -> float:
        return float(self.state.alpha)


# =============================================================================
# JIT-compiled helpers
# =============================================================================


@jax.jit
def _select_action_mt_deterministic(
    actor_apply_fn,
    actor_params: Params,
    obs: jax.Array,
    task_id: jax.Array,
    act_low: jax.Array,
    act_high: jax.Array,
) -> jax.Array:
    """Deterministic action selection for multi-task agent."""
    mean, _ = actor_apply_fn(actor_params, obs, task_id)
    action = deterministic_action(mean)
    return scale_action(action, act_low, act_high)


@jax.jit
def _select_action_mt_stochastic(
    actor_apply_fn,
    actor_params: Params,
    obs: jax.Array,
    task_id: jax.Array,
    key: PRNGKey,
    act_low: jax.Array,
    act_high: jax.Array,
) -> jax.Array:
    """Stochastic action selection for multi-task agent."""
    mean, log_std = actor_apply_fn(actor_params, obs, task_id)
    action, _ = sample_action(mean, log_std, key)
    return scale_action(action, act_low, act_high)


def _mtsac_train_step(
    state: MTSACTrainState,
    batch: Batch,
    key: PRNGKey,
    gamma: float,
    tau: float,
    target_entropy: float,
    auto_alpha: bool,
    alpha_min: float,
    alpha_max: float,
    num_tasks: int,
    obs_dim: int,
) -> Tuple[MTSACTrainState, Metrics]:
    """Multi-task SAC training step."""
    key, actor_key, target_key = jax.random.split(key, 3)
    
    # Split observations
    base_obs = batch.obs[:, :obs_dim]
    task_id = batch.obs[:, obs_dim:]
    base_next_obs = batch.next_obs[:, :obs_dim]
    next_task_id = batch.next_obs[:, obs_dim:]
    
    alpha = state.alpha
    
    # Critic update
    def critic_loss_fn(critic_params):
        q1, q2 = state.critic_apply_fn(
            critic_params, base_obs, batch.actions, task_id
        )
        
        next_mean, next_log_std = state.actor_apply_fn(
            state.actor_params, base_next_obs, next_task_id
        )
        next_action, next_log_prob = sample_action(next_mean, next_log_std, target_key)
        
        target_q1, target_q2 = state.critic_apply_fn(
            state.target_critic_params, base_next_obs, next_action, next_task_id
        )
        target_q = jnp.minimum(target_q1, target_q2) - alpha * next_log_prob
        target_q = batch.rewards + gamma * (1.0 - batch.dones) * target_q
        target_q = jax.lax.stop_gradient(target_q)
        
        q1_loss = jnp.mean((q1 - target_q) ** 2)
        q2_loss = jnp.mean((q2 - target_q) ** 2)
        return q1_loss + q2_loss, {"critic_loss": q1_loss + q2_loss, "q1": jnp.mean(q1)}
    
    critic_grads, critic_metrics = jax.grad(critic_loss_fn, has_aux=True)(state.critic_params)
    updates, new_opt_state = state.critic_optimizer.update(
        critic_grads, state.critic_opt_state, state.critic_params
    )
    new_critic_params = optax.apply_updates(state.critic_params, updates)
    state = state.replace(critic_params=new_critic_params, critic_opt_state=new_opt_state)
    
    # Actor update
    def actor_loss_fn(actor_params):
        mean, log_std = state.actor_apply_fn(actor_params, base_obs, task_id)
        action, log_prob = sample_action(mean, log_std, actor_key)
        
        q1, q2 = state.critic_apply_fn(state.critic_params, base_obs, action, task_id)
        q_min = jnp.minimum(q1, q2)
        
        actor_loss = jnp.mean(alpha * log_prob - q_min)
        return actor_loss, (log_prob, {"actor_loss": actor_loss, "entropy": -jnp.mean(log_prob)})
    
    actor_grads, (log_prob, actor_metrics) = jax.grad(actor_loss_fn, has_aux=True)(state.actor_params)
    updates, new_opt_state = state.actor_optimizer.update(
        actor_grads, state.actor_opt_state, state.actor_params
    )
    new_actor_params = optax.apply_updates(state.actor_params, updates)
    state = state.replace(actor_params=new_actor_params, actor_opt_state=new_opt_state)
    
    # Alpha update
    if auto_alpha:
        def alpha_loss_fn(log_alpha):
            return -jnp.mean(jnp.exp(log_alpha) * (jax.lax.stop_gradient(log_prob) + target_entropy))
        
        alpha_grads = jax.grad(alpha_loss_fn)(state.log_alpha)
        updates, new_opt_state = state.alpha_optimizer.update(
            alpha_grads, state.alpha_opt_state
        )
        new_log_alpha = optax.apply_updates(state.log_alpha, updates)
        new_log_alpha = jnp.clip(new_log_alpha, jnp.log(alpha_min), jnp.log(alpha_max))
        state = state.replace(log_alpha=new_log_alpha, alpha_opt_state=new_opt_state)
    
    # Target update
    state = state.soft_update_target(tau)
    state = state.replace(step=state.step + 1)
    
    metrics = {**critic_metrics, **actor_metrics, "alpha": state.alpha, "step": state.step}
    return state, metrics
