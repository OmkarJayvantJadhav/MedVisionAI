"""Convert the ISIC 2019 training release into the class-folder layout.

Inputs (from https://challenge.isic-archive.com/data/, licence CC BY-NC):

* ``ISIC_2019_Training_Input/``: the JPEG images,
* ``ISIC_2019_Training_GroundTruth.csv``: one-hot diagnosis per image,
* ``ISIC_2019_Training_Metadata.csv``: includes ``lesion_id``.

Output: ``<dest>/<Class Name>/<image>.jpg`` plus ``<dest>/groups.csv`` mapping
each image to its lesion, so photos of the same lesion stay in one split.
"""

from __future__ import annotations

import csv
import logging
import os
import shutil
from collections import Counter
from pathlib import Path

from medvision_ml.data.groups import write_groups

LOGGER = logging.getLogger(__name__)

# ISIC abbreviation -> folder (class) name. UNK has no training images.
CLASS_NAMES: dict[str, str] = {
    "AK": "Actinic Keratosis",
    "BCC": "Basal Cell Carcinoma",
    "BKL": "Benign Keratosis",
    "DF": "Dermatofibroma",
    "MEL": "Melanoma",
    "NV": "Melanocytic Nevus",
    "SCC": "Squamous Cell Carcinoma",
    "VASC": "Vascular Lesion",
}
GROUPS_FILE = "groups.csv"


def read_labels(ground_truth: Path) -> dict[str, str]:
    """Return ``image id -> class name`` from the one-hot ground-truth CSV.

    Raises:
        ValueError: If a row does not have exactly one known positive label.
    """
    labels: dict[str, str] = {}
    with ground_truth.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            positives = [
                code for code, value in row.items() if code != "image" and float(value) == 1.0
            ]
            if len(positives) != 1 or positives[0] not in CLASS_NAMES:
                raise ValueError(f"Unexpected labels for {row['image']}: {positives}")
            labels[row["image"]] = CLASS_NAMES[positives[0]]
    return labels


def read_lesions(metadata: Path) -> dict[str, str]:
    """Return ``image id -> lesion id`` for images that have one."""
    with metadata.open(newline="", encoding="utf-8") as handle:
        return {
            row["image"]: row["lesion_id"] for row in csv.DictReader(handle) if row.get("lesion_id")
        }


def _place(source: Path, target: Path, link: bool) -> None:
    if link:
        try:
            os.link(source, target)
            return
        except OSError:
            pass  # different filesystem: fall back to copying
    shutil.copy2(source, target)


def prepare_isic2019(
    images_dir: Path,
    ground_truth: Path,
    metadata: Path,
    dest: Path,
    *,
    link: bool = True,
) -> dict[str, int]:
    """Lay out ISIC 2019 images by class and write the lesion groups file.

    Args:
        images_dir: Folder with ``<image id>.jpg`` files.
        ground_truth: ``ISIC_2019_Training_GroundTruth.csv``.
        metadata: ``ISIC_2019_Training_Metadata.csv``.
        dest: Output folder; must be empty or absent.
        link: Hard-link instead of copying when possible (saves ~9 GB).

    Returns:
        Image count per class.

    Raises:
        FileExistsError: If ``dest`` is not empty.
        FileNotFoundError: If a labelled image is missing.
    """
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"Destination is not empty: {dest}")
    labels = read_labels(ground_truth)
    lesions = read_lesions(metadata)
    counts: Counter[str] = Counter()
    groups: dict[str, str] = {}
    for image_id, class_name in sorted(labels.items()):
        source = images_dir / f"{image_id}.jpg"
        if not source.is_file():
            raise FileNotFoundError(f"Missing image for {image_id}: {source}")
        target_dir = dest / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        _place(source, target_dir / source.name, link)
        groups[source.name] = lesions.get(image_id, f"image:{image_id}")
        counts[class_name] += 1
    write_groups(groups, dest / GROUPS_FILE)
    LOGGER.info("Prepared %d ISIC 2019 images in %s", sum(counts.values()), dest)
    return dict(sorted(counts.items()))
