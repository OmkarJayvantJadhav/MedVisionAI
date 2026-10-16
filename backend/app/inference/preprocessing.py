"""Image decoding, validation and the exact resize used during training."""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps, UnidentifiedImageError

from app.inference.errors import ImageTooLargeError, InvalidImageError, UnsupportedImageTypeError

ALLOWED_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})


@dataclass(frozen=True)
class ImageLimits:
    max_bytes: int = 8 * 1024 * 1024
    max_side: int = 4096
    min_side: int = 32


DEFAULT_LIMITS = ImageLimits()


def decode_image(data: bytes, limits: ImageLimits = DEFAULT_LIMITS) -> NDArray[np.uint8]:
    """Decode an uploaded image to an RGB ``uint8`` array.

    The format is detected from the content, never the file name. EXIF
    orientation is applied so phone photos are upright.

    Raises:
        ImageTooLargeError: If the file or its dimensions exceed ``limits``.
        UnsupportedImageTypeError: If it is not JPEG, PNG or WebP.
        InvalidImageError: If it cannot be decoded or is too small.
    """
    if len(data) > limits.max_bytes:
        raise ImageTooLargeError(f"Image exceeds {limits.max_bytes // (1024 * 1024)} MB.")
    if not data:
        raise InvalidImageError("The uploaded file is empty.")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in ALLOWED_FORMATS:
                raise UnsupportedImageTypeError("Only JPEG, PNG and WebP images are supported.")
            # Check dimensions from the header before decoding pixels.
            if max(image.size) > limits.max_side:
                raise ImageTooLargeError(f"Image sides must be at most {limits.max_side} px.")
            if min(image.size) < limits.min_side:
                raise InvalidImageError(f"Image sides must be at least {limits.min_side} px.")
            image.load()
            upright = ImageOps.exif_transpose(image)
            rgb = upright.convert("RGB")
            return np.asarray(rgb, dtype=np.uint8)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        raise InvalidImageError("The file could not be read as an image.") from error


def _axis_weights(
    in_size: int, out_size: int
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float32]]:
    scale = in_size / out_size
    centres = (np.arange(out_size, dtype=np.float64) + 0.5) * scale - 0.5
    floor = np.floor(centres)
    lower = np.clip(floor, 0, in_size - 1).astype(np.intp)
    upper = np.clip(np.ceil(centres), 0, in_size - 1).astype(np.intp)
    frac = (centres - floor).astype(np.float32)
    return lower, upper, frac


def resize_bilinear(image: NDArray[np.generic], height: int, width: int) -> NDArray[np.float32]:
    """Resize like ``tf.image.resize(..., method="bilinear")`` (half-pixel centres, no antialias).

    This is the resize the training pipeline applies, so serving inputs match
    what the model saw during training.
    """
    source = np.asarray(image, dtype=np.float32)
    if source.ndim == 2:
        source = source[..., np.newaxis]
    y0, y1, fy = _axis_weights(source.shape[0], height)
    x0, x1, fx = _axis_weights(source.shape[1], width)
    top = source[y0][:, x0] + (source[y0][:, x1] - source[y0][:, x0]) * fx[None, :, None]
    bottom = source[y1][:, x0] + (source[y1][:, x1] - source[y1][:, x0]) * fx[None, :, None]
    result = top + (bottom - top) * fy[:, None, None]
    return result if np.ndim(image) == 3 else result[..., 0]


def prepare_input(image: NDArray[np.uint8], size: int) -> NDArray[np.float32]:
    """Return a ``(1, size, size, 3)`` float32 batch with raw 0-255 values."""
    return resize_bilinear(image, size, size)[np.newaxis]


def display_image(image: NDArray[np.uint8], max_side: int = 512) -> NDArray[np.uint8]:
    """Downscale an image for display so its longest side is at most ``max_side``."""
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale == 1.0:
        return image
    resized = Image.fromarray(image).resize(
        (max(1, round(width * scale)), max(1, round(height * scale))), Image.Resampling.LANCZOS
    )
    return np.asarray(resized, dtype=np.uint8)
