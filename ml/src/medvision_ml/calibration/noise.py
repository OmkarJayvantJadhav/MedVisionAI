"""Sensor-noise estimate used to reject grainy images.

Immerkaer's estimator (1996) convolves the luma with a mask that cancels smooth
gradients, so what remains is noise plus texture. Dermoscopy images contain
hair and pigment texture, so the estimate is taken per 8x8 block and the 5th
percentile across blocks is kept: the smoothest regions, where texture is
absent and noise is not. Keeping the quietest blocks reads about 0.73 of the
true deviation of pure Gaussian noise; that is harmless because the threshold
is fitted on the same scale. The backend computes the same value
(``app.inference.quality.noise_level``); a parity test keeps them equal.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

BLOCK = 8
PERCENTILE = 5.0
_LUMA = np.array([0.299, 0.587, 0.114])


def noise_level(images: Any) -> NDArray[np.float64]:
    """Noise estimate (grey levels, read about 27% low) of each image in an (n, h, w, 3) batch."""
    gray = np.asarray(images, dtype=np.float64) @ _LUMA
    residual = (
        gray[:, :-2, :-2]
        - 2 * gray[:, :-2, 1:-1]
        + gray[:, :-2, 2:]
        - 2 * gray[:, 1:-1, :-2]
        + 4 * gray[:, 1:-1, 1:-1]
        - 2 * gray[:, 1:-1, 2:]
        + gray[:, 2:, :-2]
        - 2 * gray[:, 2:, 1:-1]
        + gray[:, 2:, 2:]
    )
    n, height, width = residual.shape
    rows, cols = height // BLOCK, width // BLOCK
    if rows == 0 or cols == 0:
        raise ValueError(f"Images must be at least {BLOCK + 2} pixels on each side.")
    blocks = np.abs(residual[:, : rows * BLOCK, : cols * BLOCK])
    means = blocks.reshape(n, rows, BLOCK, cols, BLOCK).mean(axis=(2, 4)).reshape(n, -1)
    result: NDArray[np.float64] = np.sqrt(np.pi / 2) / 6 * np.percentile(means, PERCENTILE, axis=1)
    return result
