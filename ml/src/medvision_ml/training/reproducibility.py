"""Seeding for reproducible runs."""

from __future__ import annotations

import os
import random

import keras
import numpy as np


def set_seed(seed: int, *, deterministic_ops: bool = False) -> None:
    """Seed Python, NumPy and the Keras/TensorFlow backends.

    Args:
        seed: Non-negative integer seed.
        deterministic_ops: Also force deterministic TensorFlow kernels. Slower,
            and some GPU ops do not support it.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    if deterministic_ops:
        import tensorflow as tf

        tf.config.experimental.enable_op_determinism()
