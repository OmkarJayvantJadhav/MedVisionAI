"""Simple image-quality checks that catch unusable photos before classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

NOISE_BLOCK = 8
NOISE_PERCENTILE = 5.0


@dataclass(frozen=True)
class QualityReport:
    brightness: float
    sharpness: float
    noise: float = 0.0


def grayscale(image: NDArray[np.floating] | NDArray[np.integer]) -> NDArray[np.float64]:
    """ITU-R BT.601 luma of an RGB image."""
    rgb = np.asarray(image, dtype=np.float64)
    return np.asarray(rgb @ np.array([0.299, 0.587, 0.114]))


def laplacian_variance(gray: NDArray[np.float64]) -> float:
    """Variance of the 4-neighbour Laplacian; low values indicate a blurry image."""
    lap = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1]
    return float(lap.var())


def noise_level(gray: NDArray[np.float64]) -> float:
    """Sensor-noise estimate in grey levels (reads about 27% low for Gaussian noise).

    Immerkaer's mask cancels smooth gradients; the mean absolute response is
    taken per 8x8 block and the 5th percentile across blocks is kept, so hair
    and pigment texture do not count as noise. Must match
    ``medvision_ml.calibration.noise.noise_level``, where the threshold is fitted.
    """
    residual = (
        gray[:-2, :-2]
        - 2 * gray[:-2, 1:-1]
        + gray[:-2, 2:]
        - 2 * gray[1:-1, :-2]
        + 4 * gray[1:-1, 1:-1]
        - 2 * gray[1:-1, 2:]
        + gray[2:, :-2]
        - 2 * gray[2:, 1:-1]
        + gray[2:, 2:]
    )
    rows, cols = residual.shape[0] // NOISE_BLOCK, residual.shape[1] // NOISE_BLOCK
    if rows == 0 or cols == 0:
        return 0.0
    blocks = np.abs(residual[: rows * NOISE_BLOCK, : cols * NOISE_BLOCK])
    means = blocks.reshape(rows, NOISE_BLOCK, cols, NOISE_BLOCK).mean(axis=(1, 3))
    return float(np.sqrt(np.pi / 2) / 6 * np.percentile(means, NOISE_PERCENTILE))


def assess(model_input: NDArray[np.floating]) -> QualityReport:
    """Measure brightness (mean luma, 0-255), sharpness and noise of a model-sized RGB image."""
    gray = grayscale(model_input)
    return QualityReport(
        brightness=float(gray.mean()),
        sharpness=laplacian_variance(gray),
        noise=noise_level(gray),
    )
