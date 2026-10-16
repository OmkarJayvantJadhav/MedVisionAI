"""Grad-CAM heatmaps computed from feature maps and head weights with NumPy.

For a ``global-average-pool -> Dense`` head, Grad-CAM for class ``c`` taken on
the pre-softmax logit equals ``ReLU(sum_k W[k, c] * A_k)`` up to a positive
constant, so no gradients are needed at serving time.
"""

from __future__ import annotations

import io

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from app.inference.preprocessing import resize_bilinear


def cam_from_features(
    feature_maps: NDArray[np.floating], kernel: NDArray[np.floating], class_index: int
) -> NDArray[np.float32]:
    """Return a ``(h, w)`` heatmap in [0, 1] from one image's ``(h, w, K)`` feature maps."""
    cam = np.tensordot(
        feature_maps.astype(np.float32), kernel[:, class_index].astype(np.float32), axes=([2], [0])
    )
    cam = np.maximum(cam, 0.0)
    peak = float(cam.max())
    return np.asarray(cam / peak if peak > 0 else np.zeros_like(cam), dtype=np.float32)


def jet(values: NDArray[np.floating]) -> NDArray[np.uint8]:
    """Map values in [0, 1] to RGB with the classic "jet" colour map."""
    v = np.clip(values, 0.0, 1.0)[..., np.newaxis]
    offsets = np.array([3.0, 2.0, 1.0])  # red, green, blue peaks
    rgb = np.clip(1.5 - np.abs(4.0 * v - offsets), 0.0, 1.0)
    return np.asarray(np.round(rgb * 255), dtype=np.uint8)


def render_heatmap(heatmap: NDArray[np.floating], height: int, width: int) -> NDArray[np.uint8]:
    """Upsample a heatmap to ``(height, width)`` and colour it."""
    return jet(resize_bilinear(heatmap, height, width))


def blend(
    image: NDArray[np.uint8], coloured: NDArray[np.uint8], alpha: float = 0.4
) -> NDArray[np.uint8]:
    """Alpha-blend a coloured heatmap over an image of the same size."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be within [0, 1].")
    mixed = (1 - alpha) * image.astype(np.float32) + alpha * coloured.astype(np.float32)
    return np.asarray(np.clip(np.round(mixed), 0, 255), dtype=np.uint8)


def encode_png(image: NDArray[np.uint8]) -> bytes:
    """Encode an RGB array as PNG bytes."""
    buffer = io.BytesIO()
    # Fast compression: optimize=True costs ~10x the time for a few percent smaller files.
    Image.fromarray(image).save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()
