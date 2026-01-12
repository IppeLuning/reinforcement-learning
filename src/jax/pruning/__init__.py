"""Pruning utilities for Lottery Ticket Hypothesis experiments.

DISCLAIMER: This code was written by Claude Opus 4.5 on 2026-01-12
and reviewed by Marinus van den Ende.

Import directly from submodules:
    from src.jax.pruning.saliency import compute_saliency, prune_by_saliency
    from src.jax.pruning.masks import MaskManager, union_masks
    from src.jax.pruning.analysis import jaccard_similarity, compute_structural_metrics
"""
