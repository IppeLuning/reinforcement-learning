"""
Recovery and Rewinding Utilities for Lottery Ticket Hypothesis.

This module handles the critical "Reset" step of LTH:
1. Loading the specific initialization weights (W0) from disk.
2. Promoting a standard SACTrainState to a MaskedTrainState.
3. Applying the binary mask to the W0 weights (W_ticket = W0 * Mask).
"""

import os
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax.core import FrozenDict

from src.agents.sac import SACAgent
from src.training.train_state import MaskedTrainState, SACTrainState
from src.utils.checkpointing import Checkpointer


def apply_mask_to_params(params: FrozenDict, mask: FrozenDict) -> FrozenDict:
    """
    Element-wise multiplication of parameters and binary mask.
    Enforces W_new = W_old * M.
    """
    return jax.tree.map(lambda p, m: p * m, params, mask)


def rewind_to_ticket(
    agent: SACAgent,
    dense_ckpt_dir: str,
    actor_mask: FrozenDict,
    critic_mask: FrozenDict,
    w0_filename: str = "checkpoint_step_20000.pkl",
) -> None:
    """
    Rewinds the agent to its initialization (W0) and applies sparsity masks.

    This function modifies the 'agent.state' in-place:
    1. Loads W0 from the dense checkpoint directory.
    2. Multiplies W0 * Mask.
    3. Converts agent.state to MaskedTrainState.
    4. Registers the masks so 'use_masking=True' works correctly.

    Args:
        agent: The initialized SACAgent (currently with random weights).
        dense_ckpt_dir: Path to the directory containing 'checkpoint_init'.
        actor_mask: Binary mask for the actor.
        critic_mask: Binary mask for the critic.
        w0_filename: Name of the W0 checkpoint file (default: "checkpoint_init").

    Raises:
        FileNotFoundError: If W0 checkpoint cannot be found.
    """
    print(f"  [LTH Recovery] Rewinding to W0 from: {dense_ckpt_dir}")

    # 1. Load W0 (The Initialization)
    checkpointer = Checkpointer(dense_ckpt_dir)

    # We load into the current agent structure just to get the shapes right,
    # then we will modify it.
    w0_state = checkpointer.restore(agent.state, item=w0_filename)

    if w0_state is None:
        raise FileNotFoundError(
            f"Could not find W0 weights ('{w0_filename}') in {dense_ckpt_dir}.\n"
            "Did you run train_dense.py? It should save W0 at step 0."
        )

    print(f"  [LTH Recovery] Loaded W0 (Step {w0_state.step})")

    # 2. Apply Masks to W0 (The "Pruning" Step)
    # Even if W0 was random, we must zero out the weights that will be pruned
    # so the agent starts exactly on the manifold defined by the mask.
    masked_actor_params = apply_mask_to_params(w0_state.actor_params, actor_mask)
    masked_critic_params = apply_mask_to_params(w0_state.critic_params, critic_mask)

    # 3. Handle Target Critic
    # The target critic typically starts as a copy of the critic.
    # We must mask it too.
    masked_target_critic_params = apply_mask_to_params(
        w0_state.target_critic_params, critic_mask
    )

    # 4. Construct the Masked State
    # We take the Optimizers from W0 (to reset momentum to 0) and the Params from W0 (masked).
    masked_state = MaskedTrainState(
        step=w0_state.step,
        # Actor
        actor_params=masked_actor_params,
        actor_opt_state=w0_state.actor_opt_state,  # Reset optimizer state
        # Critic
        critic_params=masked_critic_params,
        critic_opt_state=w0_state.critic_opt_state,  # Reset optimizer state
        target_critic_params=masked_target_critic_params,
        # Alpha (Temperature) - Usually not pruned/masked
        log_alpha=w0_state.log_alpha,
        alpha_opt_state=w0_state.alpha_opt_state,
        # Static Functions
        actor_apply_fn=w0_state.actor_apply_fn,
        critic_apply_fn=w0_state.critic_apply_fn,
        actor_optimizer=w0_state.actor_optimizer,
        critic_optimizer=w0_state.critic_optimizer,
        alpha_optimizer=w0_state.alpha_optimizer,
        # The Masks
        actor_mask=actor_mask,
        critic_mask=critic_mask,
    )

    # 5. Inject back into Agent
    agent.state = masked_state

    # Double-check: ensure the agent knows it has masks (for logic outside the state)
    # This calls the method we defined in SACAgent to ensure type consistency
    # (though MaskedTrainState already holds them, this is a safety wrapper).
    if hasattr(agent, "set_masks"):
        agent.set_masks(actor_mask, critic_mask)

    print(f"  [LTH Recovery] Agent successfully rewound and masked.")
