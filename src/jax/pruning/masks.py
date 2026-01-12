"""Mask management utilities for Lottery Ticket experiments.

DISCLAIMER: This code was written by Claude Opus 4.5 on 2026-01-12
and reviewed by Marinus van den Ende.

This module provides tools for creating, manipulating, and analyzing
binary masks that define sparse subnetworks (winning tickets).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax.core import FrozenDict

from src.jax.utils.types import Params, Mask


def create_ones_mask(params: Params) -> Mask:
    """Create a mask of all ones (no pruning).
    
    Args:
        params: Parameters to create mask for (used for structure).
        
    Returns:
        Mask with all values set to 1.0.
    """
    return jax.tree.map(jnp.ones_like, params)


def create_random_mask(
    params: Params,
    sparsity: float,
    key: jax.Array,
) -> Mask:
    """Create a random mask with given sparsity.
    
    Args:
        params: Parameters to create mask for.
        sparsity: Fraction of weights to prune (0.0 to 1.0).
        key: JAX random key.
        
    Returns:
        Random binary mask.
    """
    flat_params, tree_def = jax.tree.flatten(params)
    keys = jax.random.split(key, len(flat_params))
    
    flat_masks = []
    for param, k in zip(flat_params, keys):
        # Random values, threshold at sparsity
        rand = jax.random.uniform(k, shape=param.shape)
        mask = (rand >= sparsity).astype(jnp.float32)
        flat_masks.append(mask)
    
    return jax.tree.unflatten(tree_def, flat_masks)


def apply_mask(params: Params, mask: Mask) -> Params:
    """Apply binary mask to parameters.
    
    Args:
        params: Network parameters.
        mask: Binary mask (1 = keep, 0 = prune).
        
    Returns:
        Masked parameters with pruned weights set to 0.
    """
    return jax.tree.map(lambda p, m: p * m, params, mask)


def compute_sparsity(mask: Mask) -> float:
    """Compute the sparsity of a mask.
    
    Args:
        mask: Binary mask.
        
    Returns:
        Fraction of weights that are pruned (0.0 to 1.0).
    """
    flat_mask, _ = jax.tree.flatten(mask)
    total_params = sum(m.size for m in flat_mask)
    nonzero_params = sum(jnp.sum(m).item() for m in flat_mask)
    return 1.0 - (nonzero_params / total_params)


def compute_layerwise_sparsity(mask: Mask) -> Dict[str, float]:
    """Compute sparsity for each layer in the mask.
    
    Args:
        mask: Binary mask (nested dict structure).
        
    Returns:
        Dictionary mapping layer paths to their sparsity values.
    """
    def _flatten_with_paths(tree, prefix=""):
        results = {}
        if isinstance(tree, (dict, FrozenDict)):
            for key, value in tree.items():
                new_prefix = f"{prefix}/{key}" if prefix else key
                results.update(_flatten_with_paths(value, new_prefix))
        else:
            # Leaf node (array)
            total = tree.size
            nonzero = jnp.sum(tree).item()
            results[prefix] = 1.0 - (nonzero / total) if total > 0 else 0.0
        return results
    
    return _flatten_with_paths(mask)


def union_masks(masks: List[Mask]) -> Mask:
    """Compute union of multiple masks.
    
    A parameter is kept if it is active in ANY of the input masks.
    This is used to create the union-based multi-task subnetwork
    from single-task tickets.
    
    Args:
        masks: List of masks to union.
        
    Returns:
        Union mask where each element is max(m1, m2, ..., mn).
    """
    if len(masks) == 0:
        raise ValueError("Cannot compute union of empty mask list")
    
    if len(masks) == 1:
        return masks[0]
    
    # Union is element-wise maximum (OR for binary masks)
    result = masks[0]
    for mask in masks[1:]:
        result = jax.tree.map(jnp.maximum, result, mask)
    
    return result


def intersection_masks(masks: List[Mask]) -> Mask:
    """Compute intersection of multiple masks.
    
    A parameter is kept only if it is active in ALL input masks.
    This reveals the "shared core" across tasks.
    
    Args:
        masks: List of masks to intersect.
        
    Returns:
        Intersection mask where each element is min(m1, m2, ..., mn).
    """
    if len(masks) == 0:
        raise ValueError("Cannot compute intersection of empty mask list")
    
    if len(masks) == 1:
        return masks[0]
    
    # Intersection is element-wise minimum (AND for binary masks)
    result = masks[0]
    for mask in masks[1:]:
        result = jax.tree.map(jnp.minimum, result, mask)
    
    return result


def symmetric_difference_masks(mask1: Mask, mask2: Mask) -> Mask:
    """Compute symmetric difference of two masks.
    
    Returns elements that are in one mask but not the other.
    
    Args:
        mask1: First mask.
        mask2: Second mask.
        
    Returns:
        Symmetric difference mask.
    """
    return jax.tree.map(
        lambda m1, m2: jnp.abs(m1 - m2),
        mask1,
        mask2,
    )


@dataclass
class MaskManager:
    """Manager for storing and manipulating task-specific masks.
    
    This class provides a convenient interface for Lottery Ticket experiments,
    handling mask storage, retrieval, and set operations across tasks.
    
    Attributes:
        task_masks: Dictionary mapping task names to their masks.
        initial_params: Original random initialization (for rewinding).
        
    Example:
        >>> manager = MaskManager()
        >>> manager.store_task_mask("push-v3", actor_mask, critic_mask)
        >>> manager.store_task_mask("reach-v3", actor_mask2, critic_mask2)
        >>> union_actor, union_critic = manager.get_union_mask()
    """
    
    task_masks: Dict[str, Tuple[Mask, Mask]] = field(default_factory=dict)
    initial_actor_params: Optional[Params] = None
    initial_critic_params: Optional[Params] = None
    
    def store_initial_params(
        self,
        actor_params: Params,
        critic_params: Params,
    ) -> None:
        """Store initial parameters for weight rewinding.
        
        Args:
            actor_params: Initial actor parameters.
            critic_params: Initial critic parameters.
        """
        # Deep copy to prevent mutation
        self.initial_actor_params = jax.tree.map(jnp.copy, actor_params)
        self.initial_critic_params = jax.tree.map(jnp.copy, critic_params)
    
    def store_task_mask(
        self,
        task_name: str,
        actor_mask: Mask,
        critic_mask: Mask,
    ) -> None:
        """Store masks for a specific task.
        
        Args:
            task_name: Name of the task (e.g., "push-v3").
            actor_mask: Actor network mask.
            critic_mask: Critic network mask.
        """
        self.task_masks[task_name] = (actor_mask, critic_mask)
    
    def get_task_mask(
        self,
        task_name: str,
    ) -> Tuple[Mask, Mask]:
        """Retrieve masks for a specific task.
        
        Args:
            task_name: Name of the task.
            
        Returns:
            Tuple of (actor_mask, critic_mask).
        """
        if task_name not in self.task_masks:
            raise KeyError(f"No mask stored for task '{task_name}'")
        return self.task_masks[task_name]
    
    def get_union_mask(self) -> Tuple[Mask, Mask]:
        """Compute union of all stored task masks.
        
        Returns:
            Tuple of (union_actor_mask, union_critic_mask).
        """
        if len(self.task_masks) == 0:
            raise ValueError("No masks stored")
        
        actor_masks = [m[0] for m in self.task_masks.values()]
        critic_masks = [m[1] for m in self.task_masks.values()]
        
        return union_masks(actor_masks), union_masks(critic_masks)
    
    def get_intersection_mask(self) -> Tuple[Mask, Mask]:
        """Compute intersection of all stored task masks.
        
        Returns:
            Tuple of (intersection_actor_mask, intersection_critic_mask).
        """
        if len(self.task_masks) == 0:
            raise ValueError("No masks stored")
        
        actor_masks = [m[0] for m in self.task_masks.values()]
        critic_masks = [m[1] for m in self.task_masks.values()]
        
        return intersection_masks(actor_masks), intersection_masks(critic_masks)
    
    def get_rewound_params(
        self,
        actor_mask: Optional[Mask] = None,
        critic_mask: Optional[Mask] = None,
    ) -> Tuple[Params, Params]:
        """Get initial parameters with optional masking.
        
        Used to "rewind" to original initialization for training
        with a discovered mask (Lottery Ticket procedure).
        
        Args:
            actor_mask: Optional mask to apply to actor.
            critic_mask: Optional mask to apply to critic.
            
        Returns:
            Tuple of (actor_params, critic_params).
        """
        if self.initial_actor_params is None:
            raise ValueError("Initial params not stored")
        
        actor = self.initial_actor_params
        critic = self.initial_critic_params
        
        if actor_mask is not None:
            actor = apply_mask(actor, actor_mask)
        if critic_mask is not None:
            critic = apply_mask(critic, critic_mask)
        
        return actor, critic
    
    @property
    def task_names(self) -> List[str]:
        """List of task names with stored masks."""
        return list(self.task_masks.keys())
    
    def summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all stored masks.
        
        Returns:
            Dictionary with sparsity info for each task.
        """
        summary = {}
        for task_name, (actor_mask, critic_mask) in self.task_masks.items():
            summary[task_name] = {
                "actor_sparsity": compute_sparsity(actor_mask),
                "critic_sparsity": compute_sparsity(critic_mask),
            }
        return summary
