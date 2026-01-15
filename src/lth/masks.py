"""
Core masking utilities for Transferability experiments.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from src.utils.types import Mask, Params


def create_ones_mask(params: Params) -> Mask:
    """Create a dense mask (all ones)."""
    return jax.tree.map(jnp.ones_like, params)


def apply_mask(params: Params, mask: Mask) -> Params:
    """
    Apply binary mask to parameters (Element-wise multiplication).
    This is the operation: W_new = W_old * M
    """
    return jax.tree.map(lambda p, m: p * m, params, mask)


def compute_sparsity(mask: Mask) -> float:
    """
    Compute the global sparsity (percentage of zeros).
    Used to verify your tickets.
    """
    flat_mask, _ = jax.tree.flatten(mask)
    total_params = sum(m.size for m in flat_mask)
    nonzero_params = sum(jnp.sum(m).item() for m in flat_mask)
    return 1.0 - (nonzero_params / total_params)
