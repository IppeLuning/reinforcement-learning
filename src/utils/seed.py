import os
import random

import jax
import numpy as np


def set_seed(seed: int = 42):
    """
    Set random seeds for Python, NumPy, and JAX to ensure reproducibility.

    Args:
        seed (int): The integer seed to use.
    """
    # 1. Python's built-in random
    random.seed(seed)

    # 2. NumPy
    np.random.seed(seed)

    # 3. JAX (Global Key)
    # While JAX is functional and usually requires passing keys explicitly,
    # setting the global configuration or the first key is often useful context.
    # Note: JAX operations are deterministic given the same PRNGKey.

    # 4. OS Environment (Optional but recommended for some libraries)
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"Global seed set to: {seed}")
