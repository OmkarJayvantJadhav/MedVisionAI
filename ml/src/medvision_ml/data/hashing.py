"""Exact (MD5) and perceptual (difference hash) image fingerprints."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

HASH_SIZE = 8
HASH_BITS = HASH_SIZE * HASH_SIZE


def md5_file(path: Path) -> str:
    """Return the MD5 hex digest of a file's bytes (used for exact duplicates)."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(image: Image.Image) -> int:
    """Return the 64-bit difference hash of an image.

    The image is converted to grayscale and shrunk to 9x8 pixels; each bit
    records whether a pixel is brighter than its right-hand neighbour. Resized,
    re-encoded or slightly recoloured copies of an image produce hashes within
    a small Hamming distance of each other.
    """
    gray = image.convert("L").resize((HASH_SIZE + 1, HASH_SIZE), Image.Resampling.LANCZOS)
    pixels = np.asarray(gray, dtype=np.int16)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def dhash_file(path: Path) -> int:
    """Return the difference hash of an image file."""
    with Image.open(path) as image:
        return dhash(image)


def hamming_distances(query: int, candidates: NDArray[np.uint64]) -> NDArray[np.int64]:
    """Return the Hamming distance between ``query`` and every candidate hash."""
    xor = np.bitwise_xor(candidates, np.uint64(query))
    return np.bitwise_count(xor).astype(np.int64)
