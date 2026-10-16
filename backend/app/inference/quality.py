"""Simple image-quality checks that catch unusable photos before classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class QualityReport:
    brightness: float
    sharpness: float


def grayscale(image: NDArray[np.floating] | NDArray[np.integer]) -> NDArray[np.float64]:
    """ITU-R BT.601 luma of an RGB image."""
    rgb = np.asarray(image, dtype=np.float64)
    return np.asarray(rgb @ np.array([0.299, 0.587, 0.114]))


def laplacian_variance(gray: NDArray[np.float64]) -> float:
    """Variance of the 4-neighbour Laplacian; low values indicate a blurry image."""
    lap = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1]
    return float(lap.var())


def assess(model_input: NDArray[np.floating]) -> QualityReport:
    """Measure brightness (mean luma, 0-255) and sharpness of a model-sized RGB image."""
    gray = grayscale(model_input)
    return QualityReport(brightness=float(gray.mean()), sharpness=laplacian_variance(gray))
