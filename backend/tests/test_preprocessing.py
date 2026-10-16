from __future__ import annotations

import io

import numpy as np
import pytest
from app.inference.errors import (
    ImageTooLargeError,
    InvalidImageError,
    UnsupportedImageTypeError,
)
from app.inference.preprocessing import (
    ImageLimits,
    decode_image,
    display_image,
    prepare_input,
    resize_bilinear,
)
from PIL import Image

from tests.conftest import ImageFactory, encode, textured


@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP"])
def test_supported_formats_decode_to_rgb(image_bytes: ImageFactory, fmt: str) -> None:
    array = decode_image(image_bytes(size=(80, 60), fmt=fmt))
    assert array.shape == (60, 80, 3) and array.dtype == np.uint8


def test_png_decodes_losslessly() -> None:
    original = textured((10, 120, 240))
    np.testing.assert_array_equal(decode_image(encode(original)), original)


def test_rgba_and_grayscale_are_converted_to_rgb() -> None:
    rgba = np.dstack([textured((100, 100, 100)), np.full((64, 64), 128, np.uint8)])
    assert decode_image(encode(rgba)).shape == (64, 64, 3)
    assert decode_image(encode(textured((90, 90, 90))[..., 0])).shape == (64, 64, 3)


def test_format_is_detected_from_content_not_name() -> None:
    with pytest.raises(UnsupportedImageTypeError):
        decode_image(encode(textured((1, 2, 3)), "GIF"))
    with pytest.raises(UnsupportedImageTypeError):
        decode_image(encode(textured((1, 2, 3)), "BMP"))


@pytest.mark.parametrize("data", [b"", b"definitely not an image", b"\x89PNG\r\n\x1a\nbroken"])
def test_garbage_is_rejected(data: bytes) -> None:
    with pytest.raises(InvalidImageError):
        decode_image(data)


def test_truncated_jpeg_is_rejected(image_bytes: ImageFactory) -> None:
    data = image_bytes(size=(256, 256), fmt="JPEG")
    with pytest.raises(InvalidImageError):
        decode_image(data[: len(data) // 2])


def test_size_limits(image_bytes: ImageFactory) -> None:
    with pytest.raises(ImageTooLargeError):
        decode_image(image_bytes(), ImageLimits(max_bytes=100))
    with pytest.raises(ImageTooLargeError):
        decode_image(image_bytes(size=(300, 40)), ImageLimits(max_side=256))
    with pytest.raises(InvalidImageError):
        decode_image(image_bytes(size=(40, 20)))


def test_exif_orientation_is_applied() -> None:
    image = Image.fromarray(textured((50, 150, 50), size=(80, 40)))
    exif = image.getexif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise when displayed
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif.tobytes())
    assert decode_image(buffer.getvalue()).shape == (80, 40, 3)


def test_resize_identity_and_shapes() -> None:
    array = textured((10, 20, 30), size=(32, 32)).astype(np.float32)
    np.testing.assert_allclose(resize_bilinear(array, 32, 32), array)
    assert resize_bilinear(array, 16, 24).shape == (16, 24, 3)
    assert resize_bilinear(array[..., 0], 8, 8).shape == (8, 8)


def test_exact_halving_averages_neighbouring_pixels() -> None:
    row = np.array([[0.0, 10.0, 20.0, 30.0]])
    np.testing.assert_allclose(resize_bilinear(row, 1, 2), [[5.0, 25.0]])


def test_upscaling_interpolates_and_clamps_edges() -> None:
    row = np.array([[0.0, 100.0]])
    np.testing.assert_allclose(resize_bilinear(row, 1, 4), [[0.0, 25.0, 75.0, 100.0]])


def test_prepare_input_keeps_raw_pixel_range() -> None:
    batch = prepare_input(textured((255, 0, 128), size=(100, 50)), 32)
    assert batch.shape == (1, 32, 32, 3) and batch.dtype == np.float32
    assert batch.min() >= 0.0 and batch.max() <= 255.0


def test_display_image_only_downscales() -> None:
    small = textured((1, 1, 1), size=(100, 50))
    assert display_image(small) is small
    assert display_image(textured((1, 1, 1), size=(1024, 512)), max_side=256).shape == (128, 256, 3)
