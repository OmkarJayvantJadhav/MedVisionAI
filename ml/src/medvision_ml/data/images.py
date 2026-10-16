"""Discovery of labelled images in the ``<root>/<Split>/<class>/<file>`` layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

SUPPORTED_SUFFIXES: Final = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})
SPLITS: Final = ("Train", "Validation", "Test")


@dataclass(frozen=True)
class ImageRecord:
    """One labelled image on disk."""

    path: Path
    split: str
    label: str


def is_image(path: Path) -> bool:
    """Return whether ``path`` is a file with a supported image suffix."""
    return path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES


def list_class_images(class_dir: Path) -> list[Path]:
    """Return the supported images directly inside one class folder, sorted."""
    return sorted(p for p in class_dir.iterdir() if is_image(p))


def scan_split_dataset(root: Path, splits: tuple[str, ...] = SPLITS) -> list[ImageRecord]:
    """List every image under ``root/<split>/<class>/``.

    Args:
        root: Dataset root containing one folder per split.
        splits: Split folder names to scan; missing folders raise an error.

    Returns:
        Records sorted by split order, class name and file name.

    Raises:
        FileNotFoundError: If a split folder is missing.
    """
    records: list[ImageRecord] = []
    for split in splits:
        split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")
        for class_dir in sorted(d for d in split_dir.iterdir() if d.is_dir()):
            records.extend(
                ImageRecord(path=p, split=split, label=class_dir.name)
                for p in list_class_images(class_dir)
            )
    return records


def scan_class_dataset(root: Path) -> list[ImageRecord]:
    """List every image under ``root/<class>/`` (an unsplit source dataset)."""
    if not root.is_dir():
        raise FileNotFoundError(f"Missing dataset directory: {root}")
    records: list[ImageRecord] = []
    for class_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        records.extend(
            ImageRecord(path=p, split="", label=class_dir.name)
            for p in list_class_images(class_dir)
        )
    return records
