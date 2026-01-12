"""JAX-based Reinforcement Learning components for Lottery Ticket Hypothesis research.

This package provides a complete JAX/Flax implementation of SAC (Soft Actor-Critic)
for both single-task and multi-task reinforcement learning, with support for
pruning and structural analysis of sparse subnetworks.

Modules:
    agents: SAC and multi-task SAC agent implementations
    networks: Neural network architectures (MLP, Actor, Critic)
    training: Training loops, evaluation, and state management
    pruning: Saliency-based pruning and mask operations
    buffers: Replay buffer implementations
    utils: Utilities for checkpointing, normalization, and types
"""

# Lazy imports to avoid circular dependencies
# Users should import from submodules directly:
#   from src.jax.agents.sac import SACAgent, SACConfig
#   from src.jax.agents.multi_task_sac import MTSACAgent
