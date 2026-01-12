#!/usr/bin/env python3
"""Main entry point for JAX-based LTH training pipeline.

Usage:
    # Train push-v3 with seed 0 (single task for testing)
    python main_jax.py --task push-v3 --seed 0
    
    # Full pipeline with all tasks
    python main_jax.py --config config.yaml
    
    # Quick test run (1000 steps)
    python main_jax.py --task push-v3 --seed 0 --test
"""

import argparse
import yaml
from scripts.train_lth_pipeline import main as run_pipeline


if __name__ == "__main__":
    run_pipeline()
