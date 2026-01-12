"""Orbax-based checkpointing for training state and masks.

This module provides utilities for saving and restoring training state,
including support for mask checkpointing needed for Lottery Ticket experiments.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import jax
import orbax.checkpoint as ocp
from flax.core import FrozenDict

from src.jax.training.train_state import SACTrainState, MaskedTrainState


class Checkpointer:
    """Checkpointer for SAC training state using Orbax.
    
    Handles saving and loading of:
    - Network parameters (actor, critic, target)
    - Optimizer states
    - Learnable alpha
    - Binary masks (for Lottery Ticket experiments)
    - Training metadata (step, config, etc.)
    
    Attributes:
        checkpoint_dir: Directory for saving checkpoints.
        checkpointer: Orbax PyTreeCheckpointer instance.
        
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
        self.checkpointer = ocp.PyTreeCheckpointer()
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def _get_checkpoint_path(self, step: Optional[int] = None) -> str:
        """Get path for a specific checkpoint.
        
        Args:
            step: Training step number. If None, uses 'latest'.
            
        Returns:
            Full path to the checkpoint directory.
        """
        if step is None:
            return os.path.join(self.checkpoint_dir, "latest")
        return os.path.join(self.checkpoint_dir, f"step_{step}")
    
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
        # Extract saveable state (parameters and arrays only)
        checkpoint_data = {
            "step": state.step,
            "actor_params": state.actor_params,
            "critic_params": state.critic_params,
            "target_critic_params": state.target_critic_params,
            "log_alpha": state.log_alpha,
        }
        
        # Add masks if present
        if isinstance(state, MaskedTrainState):
            checkpoint_data["actor_mask"] = state.actor_mask
            checkpoint_data["critic_mask"] = state.critic_mask
        
        # Add extra data
        if extra is not None:
            checkpoint_data["extra"] = extra
        
        path = self._get_checkpoint_path(step)
        self.checkpointer.save(path, checkpoint_data)
        
        # Also save as 'latest' for easy restoration
        latest_path = self._get_checkpoint_path(None)
        self.checkpointer.save(latest_path, checkpoint_data)
        
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
        
        # Create template for restoration
        template = {
            "step": template_state.step,
            "actor_params": template_state.actor_params,
            "critic_params": template_state.critic_params,
            "target_critic_params": template_state.target_critic_params,
            "log_alpha": template_state.log_alpha,
        }
        
        if isinstance(template_state, MaskedTrainState):
            template["actor_mask"] = template_state.actor_mask
            template["critic_mask"] = template_state.critic_mask
            template["extra"] = {}
        else:
            template["extra"] = {}
        
        checkpoint_data = self.checkpointer.restore(path, item=template)
        
        # Reconstruct state with restored parameters
        restored_state = template_state.replace(
            step=checkpoint_data["step"],
            actor_params=checkpoint_data["actor_params"],
            critic_params=checkpoint_data["critic_params"],
            target_critic_params=checkpoint_data["target_critic_params"],
            log_alpha=checkpoint_data["log_alpha"],
        )
        
        # Restore masks if applicable
        if isinstance(template_state, MaskedTrainState):
            if "actor_mask" in checkpoint_data:
                restored_state = restored_state.replace(
                    actor_mask=checkpoint_data.get("actor_mask"),
                    critic_mask=checkpoint_data.get("critic_mask"),
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
            "actor_mask": actor_mask,
            "critic_mask": critic_mask,
        }
        
        path = os.path.join(self.checkpoint_dir, f"masks_{name}")
        self.checkpointer.save(path, mask_data)
        return path
    
    def load_masks(
        self,
        name: str,
        template_actor_mask: FrozenDict,
        template_critic_mask: FrozenDict,
    ) -> tuple[FrozenDict, FrozenDict]:
        """Load masks from checkpoint.
        
        Args:
            name: Name of the mask set to load.
            template_actor_mask: Template for actor mask structure.
            template_critic_mask: Template for critic mask structure.
            
        Returns:
            Tuple of (actor_mask, critic_mask).
        """
        path = os.path.join(self.checkpoint_dir, f"masks_{name}")
        
        template = {
            "actor_mask": template_actor_mask,
            "critic_mask": template_critic_mask,
        }
        
        mask_data = self.checkpointer.restore(path, item=template)
        return mask_data["actor_mask"], mask_data["critic_mask"]
