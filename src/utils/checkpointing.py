from __future__ import annotations

import os
import pickle
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np

from src.training.train_state import SACTrainState


def _params_to_numpy(params: Any) -> Any:
    """Converts JAX arrays in a PyTree to NumPy arrays for serialization.

    Args:
        params: A JAX PyTree containing parameters.

    Returns:
        A matching PyTree structure with NumPy arrays as leaves.
    """
    return jax.tree.map(lambda x: np.array(x) if hasattr(x, "shape") else x, params)


def _numpy_to_params(params: Any) -> Any:
    """Converts NumPy arrays back to JAX device arrays.

    Args:
        params: A PyTree containing NumPy arrays.

    Returns:
        A matching PyTree structure with JAX device arrays as leaves.
    """
    return jax.tree.map(
        lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x, params
    )


class Checkpointer:
    """Manages saving and restoring of SAC training states.

    This utility provides a simple interface to persist the agent's state to disk
    using pickle. It handles the specific requirements of the SACTrainState,
    including the optional actor and critic masks used in pruning experiments.

    Attributes:
        checkpoint_dir: Directory where checkpoint files are stored.
    """

    def __init__(self, checkpoint_dir: str) -> None:
        """Initializes the Checkpointer.

        Args:
            checkpoint_dir: Path to the directory for saving checkpoints.
        """
        self.checkpoint_dir: str = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(
        self,
        state: SACTrainState,
        filename: Optional[str] = None,
    ) -> str:
        """Saves the current training state to a checkpoint file.

        Args:
            state: The SACTrainState to be persisted.
            filename: Optional name for the file. If None, defaults to the step number.

        Returns:
            The absolute path to the saved checkpoint file.
        """
        checkpoint_data: Dict[str, Any] = {
            "step": int(state.step),
            "actor_params": _params_to_numpy(state.actor_params),
            "critic_params": _params_to_numpy(state.critic_params),
            "target_critic_params": _params_to_numpy(state.target_critic_params),
            "log_alpha": np.array(state.log_alpha),
        }

        if state.actor_mask is not None:
            checkpoint_data["actor_mask"] = _params_to_numpy(state.actor_mask)

        if state.critic_mask is not None:
            checkpoint_data["critic_mask"] = _params_to_numpy(state.critic_mask)

        if filename is None:
            filename = f"checkpoint_{state.step}"

        path: str = os.path.join(self.checkpoint_dir, f"{filename}.pkl")

        with open(path, "wb") as f:
            pickle.dump(checkpoint_data, f)

        print(f"    Saved checkpoint to {path}")

        return path

    def restore(
        self,
        template_state: SACTrainState,
        item: Optional[str] = None,
    ) -> Optional[SACTrainState]:
        """Restores a training state from a saved checkpoint file.

        Args:
            template_state: An existing SACTrainState to use as a structural template.
            item: The specific filename to restore (e.g., 'checkpoint_best').

        Returns:
            The restored SACTrainState, or None if the file does not exist.
        """
        if item:
            if not item.endswith(".pkl"):
                item += ".pkl"
            path = os.path.join(self.checkpoint_dir, item)
        else:
            return None

        if not os.path.exists(path):
            return None

        with open(path, "rb") as f:
            checkpoint_data: Dict[str, Any] = pickle.load(f)

        restored_state: SACTrainState = template_state.replace(
            step=checkpoint_data["step"],
            actor_params=_numpy_to_params(checkpoint_data["actor_params"]),
            critic_params=_numpy_to_params(checkpoint_data["critic_params"]),
            target_critic_params=_numpy_to_params(
                checkpoint_data["target_critic_params"]
            ),
            log_alpha=jnp.array(checkpoint_data["log_alpha"]),
        )

        if "actor_mask" in checkpoint_data:
            print(f"    Restoring masks from checkpoint...")
            restored_state = restored_state.replace(
                actor_mask=_numpy_to_params(checkpoint_data["actor_mask"]),
                critic_mask=_numpy_to_params(checkpoint_data.get("critic_mask")),
            )
        else:
            restored_state = restored_state.replace(actor_mask=None, critic_mask=None)

        return restored_state
