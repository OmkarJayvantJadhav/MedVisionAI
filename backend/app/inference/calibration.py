"""Logits, temperature scaling and out-of-distribution scores."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def pooled_features(feature_maps: NDArray[np.floating]) -> FloatArray:
    """Global-average-pool ``(N, h, w, K)`` feature maps to ``(N, K)``."""
    return np.asarray(feature_maps.astype(np.float64).mean(axis=(1, 2)))


def logits_from_features(
    feature_maps: NDArray[np.floating], kernel: NDArray[np.floating], bias: NDArray[np.floating]
) -> FloatArray:
    """Recompute head logits: global-average-pool the feature maps, then apply the Dense layer.

    ``feature_maps`` has shape ``(N, h, w, K)``; the result is ``(N, C)``.
    """
    pooled = pooled_features(feature_maps)
    return np.asarray(pooled @ kernel.astype(np.float64) + bias.astype(np.float64))


def softmax(logits: FloatArray, temperature: float = 1.0) -> FloatArray:
    """Numerically stable softmax of ``logits / temperature`` along the last axis."""
    scaled = logits / temperature
    shifted = scaled - scaled.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return np.asarray(exp / exp.sum(axis=-1, keepdims=True))


def energy_score(logits: FloatArray, temperature: float = 1.0) -> FloatArray:
    """Free energy ``-T * logsumexp(logits / T)``; higher means more out-of-distribution."""
    scaled = logits / temperature
    peak = scaled.max(axis=-1)
    return np.asarray(-temperature * (peak + np.log(np.exp(scaled - peak[..., None]).sum(axis=-1))))


def mahalanobis_distance(
    pooled: FloatArray, mean: NDArray[np.floating], precision: NDArray[np.floating]
) -> FloatArray:
    """Squared Mahalanobis distance of each row of ``pooled`` from the training features.

    Higher means less like the training images; the gate rejects an image when
    it exceeds the threshold fitted on validation data.
    """
    centred = pooled - mean
    return np.asarray(np.einsum("nd,de,ne->n", centred, precision, centred))
