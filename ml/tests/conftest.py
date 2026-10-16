from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

MakeImage = Callable[..., Path]


def smooth_image(seed: int, size: int = 96) -> Image.Image:
    """A random but smooth RGB image, so resized copies keep the same perceptual hash."""
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 256, size=(6, 6, 3), dtype=np.uint8)
    return Image.fromarray(coarse).resize((size, size), Image.Resampling.BICUBIC)


@pytest.fixture
def make_image() -> MakeImage:
    def _make(path: Path, seed: int, size: int = 96, fmt: str = "PNG") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        smooth_image(seed, size).save(path, format=fmt)
        return path

    return _make
