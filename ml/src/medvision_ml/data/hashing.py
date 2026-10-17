"""Exact (MD5) and perceptual (difference hash) image fingerprints."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

HASH_SIZE = 16
HASH_BITS = HASH_SIZE * HASH_SIZE
# Fraction of each side trimmed before hashing. Many dermoscopy images share a
# dark circular vignette; hashing the whole frame mostly encodes that border,
# so unrelated lesions collided (D37).
CROP_MARGIN = 0.2
_WORDS = HASH_BITS // 64


def md5_file(path: Path) -> str:
    """Return the MD5 hex digest of a file's bytes (used for exact duplicates)."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(image: Image.Image) -> int:
    """Return the 256-bit difference hash of the central part of an image.

    The image is converted to grayscale, cropped by ``CROP_MARGIN`` on every
    side and shrunk to 17x16 pixels; each bit records whether a pixel is
    brighter than its right-hand neighbour. Resized, re-encoded or slightly
    recoloured copies of an image produce hashes within a small Hamming
    distance of each other.
    """
    gray = image.convert("L")
    width, height = gray.size
    box = (
        round(width * CROP_MARGIN),
        round(height * CROP_MARGIN),
        round(width * (1 - CROP_MARGIN)),
        round(height * (1 - CROP_MARGIN)),
    )
    small = gray.crop(box).resize((HASH_SIZE + 1, HASH_SIZE), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hash_array(hashes: Iterable[int]) -> NDArray[np.uint64]:
    """Pack hashes into an ``(n, HASH_BITS // 64)`` array for :func:`hamming_distances`."""
    mask = (1 << 64) - 1
    rows = [[(value >> (64 * word)) & mask for word in range(_WORDS)] for value in hashes]
    return np.array(rows, dtype=np.uint64).reshape(-1, _WORDS)


def hamming_distances(query: int, candidates: NDArray[np.uint64]) -> NDArray[np.int64]:
    """Return the Hamming distance between ``query`` and every row of ``candidates``.

    Args:
        query: One hash.
        candidates: Hashes packed by :func:`hash_array`.
    """
    xor = np.bitwise_xor(candidates, hash_array([query]))
    distances: NDArray[np.int64] = np.bitwise_count(xor).sum(axis=1, dtype=np.int64)
    return distances
