"""``tf.data`` pipelines for the split dataset layout."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tensorflow as tf

from medvision_ml.config import DataConfig
from medvision_ml.data.images import list_class_images

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadedSplit:
    """A batched dataset with its class names and per-sample file paths (in order)."""

    dataset: Any
    class_names: list[str]
    file_paths: list[str]


def class_counts(split_dir: Path) -> dict[str, int]:
    """Return the number of images per class folder, sorted by class name."""
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")
    return {d.name: len(list_class_images(d)) for d in sorted(split_dir.iterdir()) if d.is_dir()}


def load_split(
    split_dir: Path,
    config: DataConfig,
    *,
    shuffle: bool,
    seed: int,
) -> LoadedSplit:
    """Load one split as a batched dataset of ``(uint8-range float images, int labels)``.

    Class indices follow the alphabetical order of the class folders, which is
    the order written to ``class_names.json``.

    Returns:
        The dataset, its class names and, when ``shuffle`` is false, the file
        path of every sample in iteration order.

    Raises:
        FileNotFoundError: If the split is missing.
        ValueError: If the split contains no images.
    """
    counts = class_counts(split_dir)
    total = sum(counts.values())
    if total == 0:
        raise ValueError(f"No images found in {split_dir}")
    # Load single images (batch_size=None) so the cache and the shuffle buffer
    # hold images, not batches: shuffling batches kept the same images together
    # every epoch and needed ~12 GB of RAM for ISIC 2019's shuffle buffer.
    dataset = tf.keras.utils.image_dataset_from_directory(
        split_dir,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=None,
        image_size=(config.image_size, config.image_size),
        shuffle=shuffle,
        seed=seed if shuffle else None,
    )
    class_names = list(dataset.class_names)
    file_paths = [] if shuffle else list(dataset.file_paths)
    if config.cache and config.cache_dir is not None:
        config.cache_dir.mkdir(parents=True, exist_ok=True)
        dataset = dataset.cache(str(config.cache_dir / split_dir.name.lower()))
    elif config.cache:
        dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(
            min(config.shuffle_buffer_size, total), seed=seed, reshuffle_each_iteration=True
        )
    dataset = dataset.batch(config.batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    LOGGER.info("Loaded %s: %d images in %d classes", split_dir, total, len(class_names))
    return LoadedSplit(dataset=dataset, class_names=class_names, file_paths=file_paths)
