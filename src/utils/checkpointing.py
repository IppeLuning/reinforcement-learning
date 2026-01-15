"""Simple checkpointing for training state using pickle/numpy."""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np

from src.training.train_state import MaskedTrainState, SACTrainState


def _params_to_numpy(params: Any) -> Any:
    """Convert JAX arrays in params tree to numpy for serialization."""
    return jax.tree.map(lambda x: np.array(x) if hasattr(x, "shape") else x, params)


def _numpy_to_params(params: Any) -> Any:
    """Convert numpy arrays back to JAX arrays."""
    return jax.tree.map(
        lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x, params
    )


class Checkpointer:
    """Simple checkpointer for SAC training state.

    Attributes:
        checkpoint_dir: Directory for saving checkpoints.
    """

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(
        self,
        state: SACTrainState,
        filename: Optional[str] = None,
    ) -> str:
        """Save training state to checkpoint."""

        checkpoint_data = {
            "step": int(state.step),
            "actor_params": _params_to_numpy(state.actor_params),
            "critic_params": _params_to_numpy(state.critic_params),
            "target_critic_params": _params_to_numpy(state.target_critic_params),
            "log_alpha": np.array(state.log_alpha),
        }

        # [CRITICAL FIX] Save masks if they exist (for resuming sparse runs)
        if isinstance(state, MaskedTrainState):
            if state.actor_mask is not None:
                checkpoint_data["actor_mask"] = _params_to_numpy(state.actor_mask)
            if state.critic_mask is not None:
                checkpoint_data["critic_mask"] = _params_to_numpy(state.critic_mask)

        # Determine path
        if filename:
            path = os.path.join(self.checkpoint_dir, f"{filename}.pkl")

        with open(path, "wb") as f:
            pickle.dump(checkpoint_data, f)

        print(f"    Saved checkpoint to {path}")

        return path

    def restore(
        self,
        template_state: SACTrainState,
        item: Optional[str] = None,  # <--- Added to support 'checkpoint_init'
    ) -> Optional[SACTrainState]:
        """Restore training state from checkpoint.

        Args:
            template_state: A template state to infer structure.
            item: Explicit filename (e.g. 'checkpoint_init').

        Returns:
            Restored state, or None if not found.
        """
        # Determine path
        if item:
            # Handles "checkpoint_init" or "checkpoint_best"
            # We append .pkl if user didn't provided it
            if not item.endswith(".pkl"):
                item += ".pkl"
            path = os.path.join(self.checkpoint_dir, item)

        if not os.path.exists(path):
            return None

        with open(path, "rb") as f:
            checkpoint_data = pickle.load(f)

        # Reconstruct state
        restored_state = template_state.replace(
            step=checkpoint_data["step"],
            actor_params=_numpy_to_params(checkpoint_data["actor_params"]),
            critic_params=_numpy_to_params(checkpoint_data["critic_params"]),
            target_critic_params=_numpy_to_params(
                checkpoint_data["target_critic_params"]
            ),
            log_alpha=jnp.array(checkpoint_data["log_alpha"]),
        )

        # [CRITICAL FIX] Restore masks if present in checkpoint
        # This allows resuming a sparse run seamlessly
        if isinstance(template_state, MaskedTrainState):
            if "actor_mask" in checkpoint_data:
                restored_state = restored_state.replace(
                    actor_mask=_numpy_to_params(checkpoint_data["actor_mask"]),
                    critic_mask=_numpy_to_params(checkpoint_data["critic_mask"]),
                )

        return restored_state
