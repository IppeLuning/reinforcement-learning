"""Simple checkpointing for training state and masks using pickle/numpy.

DISCLAIMER: This code was written by Claude Opus 4.5 on 2026-01-12
and reviewed by Marinus van den Ende.

Orbax has compatibility issues with jax-metal, so we use a simpler approach.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict

from src.jax.training.train_state import SACTrainState, MaskedTrainState


def _params_to_numpy(params: Any) -> Any:
    """Convert JAX arrays in params tree to numpy for serialization."""
    return jax.tree.map(lambda x: np.array(x) if hasattr(x, 'shape') else x, params)


def _numpy_to_params(params: Any) -> Any:
    """Convert numpy arrays back to JAX arrays."""
    return jax.tree.map(lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x, params)


class Checkpointer:
    """Simple checkpointer for SAC training state.
    
    Uses pickle for simplicity and compatibility.
    
    Attributes:
        checkpoint_dir: Directory for saving checkpoints.
        
    Example:
        >>> ckpt = Checkpointer("checkpoints/sac_run1")
        >>> ckpt.save(state, step=10000)
        >>> restored_state = ckpt.restore(state, step=10000)
    """
    
    def __init__(self, checkpoint_dir: str):
        """Initialize checkpointer.
        
        Args:
            checkpoint_dir: Directory to store checkpoints.
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def _get_checkpoint_path(self, step: Optional[int] = None) -> str:
        """Get path for a specific checkpoint."""
        if step is None:
            return os.path.join(self.checkpoint_dir, "latest.pkl")
        return os.path.join(self.checkpoint_dir, f"step_{step}.pkl")
    
    def save(
        self,
        state: SACTrainState,
        step: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save training state to checkpoint.
        
        Args:
            state: SACTrainState to save.
            step: Training step number for checkpoint naming.
            extra: Additional data to save (e.g., config, metrics).
            
        Returns:
            Path where checkpoint was saved.
        """
        # Extract saveable state (convert to numpy)
        checkpoint_data = {
            "step": int(state.step),
            "actor_params": _params_to_numpy(state.actor_params),
            "critic_params": _params_to_numpy(state.critic_params),
            "target_critic_params": _params_to_numpy(state.target_critic_params),
            "log_alpha": np.array(state.log_alpha),
        }
        
        # Add masks if present
        if isinstance(state, MaskedTrainState):
            checkpoint_data["actor_mask"] = _params_to_numpy(state.actor_mask)
            checkpoint_data["critic_mask"] = _params_to_numpy(state.critic_mask)
        
        # Add extra data
        if extra is not None:
            checkpoint_data["extra"] = extra
        
        path = self._get_checkpoint_path(step)
        with open(path, "wb") as f:
            pickle.dump(checkpoint_data, f)
        
        # Also save as 'latest' for easy restoration
        latest_path = self._get_checkpoint_path(None)
        with open(latest_path, "wb") as f:
            pickle.dump(checkpoint_data, f)
        
        return path
    
    def restore(
        self,
        template_state: SACTrainState,
        step: Optional[int] = None,
    ) -> SACTrainState:
        """Restore training state from checkpoint.
        
        Args:
            template_state: A template state with correct structure and apply_fns.
            step: Step number to restore. If None, restores 'latest'.
            
        Returns:
            Restored SACTrainState with loaded parameters.
        """
        path = self._get_checkpoint_path(step)
        
        with open(path, "rb") as f:
            checkpoint_data = pickle.load(f)
        
        # Reconstruct state with restored parameters (convert back to JAX)
        restored_state = template_state.replace(
            step=checkpoint_data["step"],
            actor_params=_numpy_to_params(checkpoint_data["actor_params"]),
            critic_params=_numpy_to_params(checkpoint_data["critic_params"]),
            target_critic_params=_numpy_to_params(checkpoint_data["target_critic_params"]),
            log_alpha=jnp.array(checkpoint_data["log_alpha"]),
        )
        
        # Restore masks if applicable
        if isinstance(template_state, MaskedTrainState):
            if "actor_mask" in checkpoint_data:
                restored_state = restored_state.replace(
                    actor_mask=_numpy_to_params(checkpoint_data.get("actor_mask")),
                    critic_mask=_numpy_to_params(checkpoint_data.get("critic_mask")),
                )
        
        return restored_state
    
    def save_masks(
        self,
        actor_mask: FrozenDict,
        critic_mask: FrozenDict,
        name: str,
    ) -> str:
        """Save masks separately for Lottery Ticket analysis.
        
        Args:
            actor_mask: Binary mask for actor network.
            critic_mask: Binary mask for critic networks.
            name: Name for this mask set (e.g., "task_push_v3").
            
        Returns:
            Path where masks were saved.
        """
        mask_data = {
            "actor_mask": _params_to_numpy(actor_mask),
            "critic_mask": _params_to_numpy(critic_mask),
        }
        
        path = os.path.join(self.checkpoint_dir, f"masks_{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(mask_data, f)
        return path
    
    def load_masks(
        self,
        name: str,
        template_actor_mask: FrozenDict = None,
        template_critic_mask: FrozenDict = None,
    ) -> tuple[FrozenDict, FrozenDict]:
        """Load masks from checkpoint.
        
        Args:
            name: Name of the mask set to load.
            template_actor_mask: Unused, for API compatibility.
            template_critic_mask: Unused, for API compatibility.
            
        Returns:
            Tuple of (actor_mask, critic_mask).
        """
        path = os.path.join(self.checkpoint_dir, f"masks_{name}.pkl")
        
        with open(path, "rb") as f:
            mask_data = pickle.load(f)
        
        return (
            _numpy_to_params(mask_data["actor_mask"]),
            _numpy_to_params(mask_data["critic_mask"]),
        )
