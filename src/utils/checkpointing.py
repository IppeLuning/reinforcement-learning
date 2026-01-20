"""Simple checkpointing for training state using pickle/numpy."""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np

from src.training.train_state import SACTrainState


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
        save_optimizer: bool = True,
    ) -> str:
        """Save training state to checkpoint.
        
        Args:
            state: The SACTrainState to save.
            filename: Optional filename (without .pkl extension).
            save_optimizer: Whether to save optimizer states for exact resumption.
                           Set to False for rewind checkpoints where we want fresh optimizers.
        """

        checkpoint_data = {
            "step": int(state.step),
            "actor_params": _params_to_numpy(state.actor_params),
            "critic_params": _params_to_numpy(state.critic_params),
            "target_critic_params": _params_to_numpy(state.target_critic_params),
            "log_alpha": np.array(state.log_alpha),
        }

        # Save optimizer states for reproducible resumption
        if save_optimizer:
            checkpoint_data["actor_opt_state"] = _params_to_numpy(state.actor_opt_state)
            checkpoint_data["critic_opt_state"] = _params_to_numpy(state.critic_opt_state)
            checkpoint_data["alpha_opt_state"] = _params_to_numpy(state.alpha_opt_state)

        # Save masks if present
        if state.actor_mask is not None:
            checkpoint_data["actor_mask"] = _params_to_numpy(state.actor_mask)

        if state.critic_mask is not None:
            checkpoint_data["critic_mask"] = _params_to_numpy(state.critic_mask)

        # Determine path
        if filename is None:
            filename = f"checkpoint_{state.step}"

        path = os.path.join(self.checkpoint_dir, f"{filename}.pkl")

        with open(path, "wb") as f:
            pickle.dump(checkpoint_data, f)

        print(f"    Saved checkpoint to {path}")

        return path

    def restore(
        self,
        template_state: SACTrainState,
        item: Optional[str] = None,
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
            if not item.endswith(".pkl"):
                item += ".pkl"
            path = os.path.join(self.checkpoint_dir, item)
        else:
            # Fallback or specific logic if needed
            return None

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

        # Restore optimizer states if present (for exact resumption)
        if "actor_opt_state" in checkpoint_data:
            print(f"    Restoring optimizer states from checkpoint...")
            restored_state = restored_state.replace(
                actor_opt_state=_numpy_to_params(checkpoint_data["actor_opt_state"]),
                critic_opt_state=_numpy_to_params(checkpoint_data["critic_opt_state"]),
                alpha_opt_state=_numpy_to_params(checkpoint_data["alpha_opt_state"]),
            )
        else:
            print(f"    Note: Optimizer states not in checkpoint, using fresh optimizers")

        # Restore masks if present in checkpoint
        if "actor_mask" in checkpoint_data:
            print(f"    Restoring masks from checkpoint...")
            restored_state = restored_state.replace(
                actor_mask=_numpy_to_params(checkpoint_data["actor_mask"]),
                critic_mask=_numpy_to_params(checkpoint_data.get("critic_mask")),
            )
        else:
            # If loading a dense checkpoint into a potentially sparse pipeline,
            # ensure we reset masks to None (or keep template's None)
            restored_state = restored_state.replace(actor_mask=None, critic_mask=None)

        return restored_state
