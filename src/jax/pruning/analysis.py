"""Structural analysis tools for comparing sparse subnetworks.

DISCLAIMER: This code was written by Claude Opus 4.5 on 2026-01-12
and reviewed by Marinus van den Ende.

This module provides metrics for analyzing the relationship between
different masks/subnetworks, including:
- Jaccard similarity (edge-level)
- Neuron overlap (unit-level)
- Layerwise sharing patterns
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
from flax.core import FrozenDict

from src.jax.utils.types import Mask


def jaccard_similarity(mask1: Mask, mask2: Mask) -> float:
    """Compute Jaccard similarity between two masks.
    
    Jaccard(A, B) = |A ∩ B| / |A ∪ B|
    
    This measures the overlap between two sparse subnetworks at the
    edge (parameter) level.
    
    Args:
        mask1: First binary mask.
        mask2: Second binary mask.
        
    Returns:
        Jaccard similarity coefficient (0.0 to 1.0).
    """
    # Flatten both masks
    flat1, _ = jax.tree.flatten(mask1)
    flat2, _ = jax.tree.flatten(mask2)
    
    all_m1 = jnp.concatenate([m.flatten() for m in flat1])
    all_m2 = jnp.concatenate([m.flatten() for m in flat2])
    
    # Compute intersection and union
    intersection = jnp.sum(jnp.minimum(all_m1, all_m2))
    union = jnp.sum(jnp.maximum(all_m1, all_m2))
    
    # Handle edge case of empty union
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return float(intersection / union)


def layerwise_jaccard(mask1: Mask, mask2: Mask) -> Dict[str, float]:
    """Compute Jaccard similarity for each layer separately.
    
    This reveals whether masks are more similar in early layers
    (shared representations) vs. later layers (task-specific).
    
    Args:
        mask1: First mask (nested dict structure).
        mask2: Second mask (same structure).
        
    Returns:
        Dictionary mapping layer paths to Jaccard similarity.
    """
    def _compute_layerwise(m1, m2, prefix=""):
        results = {}
        
        if isinstance(m1, (dict, FrozenDict)):
            for key in m1.keys():
                new_prefix = f"{prefix}/{key}" if prefix else key
                results.update(_compute_layerwise(m1[key], m2[key], new_prefix))
        else:
            # Leaf node - compute Jaccard for this layer
            flat1 = m1.flatten()
            flat2 = m2.flatten()
            
            intersection = jnp.sum(jnp.minimum(flat1, flat2))
            union = jnp.sum(jnp.maximum(flat1, flat2))
            
            if union > 0:
                results[prefix] = float(intersection / union)
            else:
                results[prefix] = 1.0 if intersection == 0 else 0.0
        
        return results
    
    return _compute_layerwise(mask1, mask2)


def neuron_overlap(
    mask1: Mask,
    mask2: Mask,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute neuron-level overlap between masks.
    
    A neuron is considered "active" if at least `threshold` fraction
    of its incoming weights are non-zero. This metric measures whether
    the same neurons are being used across different tasks.
    
    Args:
        mask1: First mask.
        mask2: Second mask.
        threshold: Fraction of weights needed to consider neuron active.
        
    Returns:
        Dictionary with neuron overlap metrics for each layer.
    """
    def _compute_neuron_overlap(m1, m2, prefix=""):
        results = {}
        
        if isinstance(m1, (dict, FrozenDict)):
            for key in m1.keys():
                new_prefix = f"{prefix}/{key}" if prefix else key
                results.update(_compute_neuron_overlap(m1[key], m2[key], new_prefix))
        else:
            # Only for 2D weight matrices (not biases)
            if m1.ndim == 2:
                # Compute fraction of incoming weights active per neuron
                # For weight matrix (in_dim, out_dim), neurons are columns
                active_frac1 = jnp.mean(m1, axis=0)
                active_frac2 = jnp.mean(m2, axis=0)
                
                # Neurons active in each mask
                neurons1 = (active_frac1 >= threshold).astype(jnp.float32)
                neurons2 = (active_frac2 >= threshold).astype(jnp.float32)
                
                # Overlap
                intersection = jnp.sum(neurons1 * neurons2)
                union = jnp.sum(jnp.maximum(neurons1, neurons2))
                
                if union > 0:
                    results[prefix] = float(intersection / union)
                else:
                    results[prefix] = 1.0
        
        return results
    
    return _compute_neuron_overlap(mask1, mask2)


def compute_structural_metrics(
    masks: Dict[str, Mask],
    reference_mask: Mask | None = None,
) -> Dict[str, Dict]:
    """Compute comprehensive structural metrics for a set of masks.
    
    This is the main analysis function for comparing single-task masks
    with each other and with a reference (e.g., universal multi-task mask).
    
    Args:
        masks: Dictionary mapping task names to their masks.
        reference_mask: Optional reference mask to compare against.
        
    Returns:
        Dictionary containing:
            - pairwise_jaccard: Jaccard between all mask pairs
            - layerwise_patterns: Layerwise similarity analysis
            - sparsity: Sparsity of each mask
            - reference_similarity: Similarity to reference (if provided)
    """
    task_names = list(masks.keys())
    results = {
        "pairwise_jaccard": {},
        "layerwise_patterns": {},
        "sparsity": {},
        "neuron_overlap": {},
    }
    
    # Compute sparsity for each mask
    for task_name, mask in masks.items():
        flat_mask, _ = jax.tree.flatten(mask)
        total = sum(m.size for m in flat_mask)
        nonzero = sum(jnp.sum(m).item() for m in flat_mask)
        results["sparsity"][task_name] = 1.0 - (nonzero / total)
    
    # Pairwise Jaccard similarity
    for i, task1 in enumerate(task_names):
        for task2 in task_names[i + 1:]:
            key = f"{task1}_vs_{task2}"
            results["pairwise_jaccard"][key] = jaccard_similarity(
                masks[task1], masks[task2]
            )
            
            # Layerwise analysis
            results["layerwise_patterns"][key] = layerwise_jaccard(
                masks[task1], masks[task2]
            )
            
            # Neuron overlap
            results["neuron_overlap"][key] = neuron_overlap(
                masks[task1], masks[task2]
            )
    
    # Reference comparison
    if reference_mask is not None:
        results["reference_similarity"] = {}
        results["reference_layerwise"] = {}
        
        for task_name, mask in masks.items():
            results["reference_similarity"][task_name] = jaccard_similarity(
                mask, reference_mask
            )
            results["reference_layerwise"][task_name] = layerwise_jaccard(
                mask, reference_mask
            )
    
    return results


def analyze_shared_core(masks: Dict[str, Mask]) -> Dict[str, float]:
    """Analyze the shared sparse core across tasks.
    
    Computes what fraction of each task's active weights are also
    active in all other tasks (the "shared core").
    
    Args:
        masks: Dictionary mapping task names to masks.
        
    Returns:
        Dictionary with shared core statistics.
    """
    task_names = list(masks.keys())
    if len(task_names) < 2:
        return {"shared_core_fraction": 1.0}
    
    # Compute intersection of all masks
    mask_list = list(masks.values())
    shared_mask = mask_list[0]
    for mask in mask_list[1:]:
        shared_mask = jax.tree.map(jnp.minimum, shared_mask, mask)
    
    # Compute union of all masks
    union_mask = mask_list[0]
    for mask in mask_list[1:]:
        union_mask = jax.tree.map(jnp.maximum, union_mask, mask)
    
    # Count parameters
    flat_shared, _ = jax.tree.flatten(shared_mask)
    flat_union, _ = jax.tree.flatten(union_mask)
    
    shared_count = sum(jnp.sum(m).item() for m in flat_shared)
    union_count = sum(jnp.sum(m).item() for m in flat_union)
    
    results = {
        "shared_core_size": shared_count,
        "union_size": union_count,
        "shared_core_fraction": shared_count / union_count if union_count > 0 else 0.0,
    }
    
    # Per-task: what fraction of task's weights are in shared core?
    for task_name, mask in masks.items():
        flat_mask, _ = jax.tree.flatten(mask)
        task_count = sum(jnp.sum(m).item() for m in flat_mask)
        results[f"{task_name}_in_shared"] = (
            shared_count / task_count if task_count > 0 else 0.0
        )
    
    return results


def generate_analysis_report(
    single_task_masks: Dict[str, Mask],
    multitask_mask: Mask | None = None,
    union_mask: Mask | None = None,
) -> str:
    """Generate a human-readable analysis report.
    
    Args:
        single_task_masks: Masks from independently trained single-task agents.
        multitask_mask: Optional mask from jointly trained multi-task agent.
        union_mask: Optional union of single-task masks.
        
    Returns:
        Formatted string report.
    """
    lines = ["=" * 60]
    lines.append("STRUCTURAL ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append("")
    
    # Sparsity analysis
    lines.append("## Sparsity Analysis")
    lines.append("-" * 40)
    for task_name, mask in single_task_masks.items():
        flat_mask, _ = jax.tree.flatten(mask)
        total = sum(m.size for m in flat_mask)
        nonzero = sum(jnp.sum(m).item() for m in flat_mask)
        sparsity = 1.0 - (nonzero / total)
        lines.append(f"  {task_name}: {sparsity:.2%} sparse")
    lines.append("")
    
    # Pairwise similarity
    lines.append("## Pairwise Jaccard Similarity")
    lines.append("-" * 40)
    task_names = list(single_task_masks.keys())
    for i, task1 in enumerate(task_names):
        for task2 in task_names[i + 1:]:
            sim = jaccard_similarity(single_task_masks[task1], single_task_masks[task2])
            lines.append(f"  {task1} vs {task2}: {sim:.4f}")
    lines.append("")
    
    # Shared core analysis
    if len(single_task_masks) >= 2:
        lines.append("## Shared Core Analysis")
        lines.append("-" * 40)
        core_stats = analyze_shared_core(single_task_masks)
        lines.append(f"  Shared core fraction: {core_stats['shared_core_fraction']:.2%}")
        for task_name in task_names:
            frac = core_stats.get(f"{task_name}_in_shared", 0.0)
            lines.append(f"  {task_name} weights in shared core: {frac:.2%}")
        lines.append("")
    
    # Comparison with multi-task mask
    if multitask_mask is not None:
        lines.append("## Comparison with Multi-Task Mask")
        lines.append("-" * 40)
        for task_name, mask in single_task_masks.items():
            sim = jaccard_similarity(mask, multitask_mask)
            lines.append(f"  {task_name} vs MT: {sim:.4f}")
        lines.append("")
    
    # Comparison with union mask
    if union_mask is not None:
        lines.append("## Comparison with Union Mask")
        lines.append("-" * 40)
        for task_name, mask in single_task_masks.items():
            sim = jaccard_similarity(mask, union_mask)
            lines.append(f"  {task_name} vs Union: {sim:.4f}")
        
        if multitask_mask is not None:
            mt_union_sim = jaccard_similarity(multitask_mask, union_mask)
            lines.append(f"  MT vs Union: {mt_union_sim:.4f}")
        lines.append("")
    
    lines.append("=" * 60)
    return "\n".join(lines)
