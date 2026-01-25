import os
import random

import jax
import numpy as np


def set_seed(seed: int = 42):
    """
    Sets seeds for Python and NumPy, and initializes the JAX key.

    Args:
        seed (int): The integer seed to use.

    Returns:
        jax.random.PRNGKey: The root random key for JAX operations.
    """
    # 1. Python random
    random.seed(seed)

    # 2. NumPy
    np.random.seed(seed)

    # 3. JAX (Stateless)
    key = jax.random.key(seed)

    # 4. OS Environment
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"Global seed set to: {seed}")

    return key
