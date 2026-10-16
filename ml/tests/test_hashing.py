from __future__ import annotations

import numpy as np
from medvision_ml.data.hashing import dhash, hamming_distances
from PIL import Image

from tests.conftest import smooth_image


def test_resized_and_reencoded_copies_have_close_hashes() -> None:
    original = smooth_image(1, 128)
    resized = original.resize((64, 64), Image.Resampling.BILINEAR)
    assert bin(dhash(original) ^ dhash(resized)).count("1") <= 4


def test_different_images_have_distant_hashes() -> None:
    distances = [
        bin(dhash(smooth_image(0)) ^ dhash(smooth_image(s))).count("1") for s in range(1, 6)
    ]
    assert min(distances) > 8


def test_hamming_distances_vectorised() -> None:
    candidates = np.array([0b0, 0b1, 0b111, 2**64 - 1], dtype=np.uint64)
    assert hamming_distances(0, candidates).tolist() == [0, 1, 3, 64]
