"""Class weights that counter class imbalance in the loss."""

from __future__ import annotations

import math
from collections.abc import Mapping

from medvision_ml.config import ClassWeighting


def compute_class_weights(
    counts: Mapping[int, int],
    scheme: ClassWeighting,
) -> dict[int, float] | None:
    """Return Keras ``class_weight`` values for the given per-class counts.

    Schemes:
        * ``none``: no weighting.
        * ``balanced``: ``n_samples / (n_classes * n_c)``; rare classes get large weights.
        * ``sqrt``: the square root of ``balanced``. The prototype found this keeps most
          of the minority-class recall gain while losing far less overall accuracy.

    Weights are rescaled so that their sample-weighted mean is 1, which keeps the
    loss on the same scale as the unweighted loss.

    Raises:
        ValueError: If any class has no samples.
    """
    if scheme == "none":
        return None
    if not counts or any(c <= 0 for c in counts.values()):
        raise ValueError("Every class needs at least one training sample.")
    total = sum(counts.values())
    balanced = {k: total / (len(counts) * c) for k, c in counts.items()}
    weights = balanced if scheme == "balanced" else {k: math.sqrt(w) for k, w in balanced.items()}
    mean = sum(weights[k] * counts[k] for k in counts) / total
    return {k: w / mean for k, w in sorted(weights.items())}
