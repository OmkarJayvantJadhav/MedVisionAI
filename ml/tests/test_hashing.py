from __future__ import annotations

import numpy as np
from medvision_ml.data.hashing import HASH_BITS, dhash, hamming_distances, hash_array
from PIL import Image, ImageDraw, ImageFilter

from tests.conftest import smooth_image


def _distance(a: Image.Image, b: Image.Image) -> int:
    return bin(dhash(a) ^ dhash(b)).count("1")


def _dermoscopy_like(seed: int, size: int = 256) -> Image.Image:
    """A small lesion on faintly textured skin inside a dark circular border."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 256, size=(32, 32), dtype=np.uint8)
    texture = np.asarray(Image.fromarray(noise).resize((size, size), Image.Resampling.BICUBIC))
    skin = np.array([200, 150, 130], dtype=np.float32) + (texture[..., None] / 255 - 0.5) * 12
    image = Image.fromarray(np.clip(skin, 0, 255).astype(np.uint8))
    cx, cy = rng.uniform(0.4, 0.6, size=2) * size
    radius = rng.uniform(0.08, 0.15) * size
    lesion = (cx - radius, cy - 0.8 * radius, cx + radius, cy + 0.8 * radius)
    ImageDraw.Draw(image).ellipse(lesion, fill=(90, 60, 50))
    image = image.filter(ImageFilter.GaussianBlur(3))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((size * 0.1, size * 0.1, size * 0.9, size * 0.9), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(size * 0.04))
    return Image.composite(image, Image.new("RGB", (size, size)), mask)


def test_hash_has_256_bits() -> None:
    assert HASH_BITS == 256
    assert dhash(smooth_image(0)) < 2**HASH_BITS


def test_resized_and_reencoded_copies_have_close_hashes() -> None:
    original = smooth_image(1, 128)
    resized = original.resize((64, 64), Image.Resampling.BILINEAR)
    assert _distance(original, resized) <= 16


def test_different_images_have_distant_hashes() -> None:
    distances = [_distance(smooth_image(0), smooth_image(s)) for s in range(1, 6)]
    assert min(distances) > 32


def test_shared_dark_border_does_not_make_images_similar() -> None:
    # Regression (D37): an 8x8 hash of the whole frame mostly encoded the
    # border, so these images were 1-7 bits apart (threshold 4) and thousands of
    # unrelated ISIC images formed one "duplicate" group.
    first = _dermoscopy_like(0)
    distances = [_distance(first, _dermoscopy_like(s)) for s in range(1, 8)]
    assert min(distances) > 32
    copy = first.resize((180, 180), Image.Resampling.BILINEAR)
    assert _distance(first, copy) <= 16


def test_hamming_distances_vectorised() -> None:
    candidates = hash_array([0b0, 0b1, 0b111, 2**64 - 1, 2**256 - 1, 1 << 200])
    assert hamming_distances(0, candidates).tolist() == [0, 1, 3, 64, 256, 1]
    assert hamming_distances(2**256 - 1, candidates).tolist() == [256, 255, 253, 192, 0, 255]
