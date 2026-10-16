"""The training pipeline must shuffle images, not whole batches.

Shuffling batches kept the same images together every epoch and made the
shuffle buffer hold ~1,000 batches (about 12 GB for ISIC 2019), which ran a
free Colab VM out of memory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
from medvision_ml.config import DataConfig
from medvision_ml.data.dataset import load_split

from tests.conftest import MakeImage

N_IMAGES = 16
BATCH = 4


def make_split(root: Path, make_image: MakeImage) -> Path:
    for i in range(N_IMAGES):
        make_image(root / ("A" if i % 2 else "B") / f"{i:02d}.png", seed=i, size=40)
    return root


def batch_compositions(dataset: tf.data.Dataset) -> list[frozenset[float]]:
    """Identify each image by its mean pixel value and return the set per batch."""
    return [
        frozenset(np.round(images.numpy().mean(axis=(1, 2, 3)), 3).tolist())
        for images, _ in dataset
    ]


def test_training_split_reshuffles_images_across_batches(
    tmp_path: Path, make_image: MakeImage
) -> None:
    config = DataConfig(image_size=32, batch_size=BATCH, cache=True, shuffle_buffer_size=64)
    split = load_split(make_split(tmp_path / "Train", make_image), config, shuffle=True, seed=1)

    epochs = [batch_compositions(split.dataset) for _ in range(3)]
    for batches in epochs:
        assert sum(len(b) for b in batches) == N_IMAGES
        assert all(len(b) == BATCH for b in batches)
    # With batch-level shuffling every epoch would contain exactly the same batches.
    assert len({frozenset(batches) for batches in epochs}) > 1


def test_evaluation_split_keeps_file_order(tmp_path: Path, make_image: MakeImage) -> None:
    config = DataConfig(image_size=32, batch_size=BATCH)
    split = load_split(make_split(tmp_path / "Test", make_image), config, shuffle=False, seed=1)
    labels = np.concatenate([y.numpy() for _, y in split.dataset])
    assert len(split.file_paths) == N_IMAGES == len(labels)
    expected = [split.class_names.index(Path(p).parent.name) for p in split.file_paths]
    assert labels.tolist() == expected
