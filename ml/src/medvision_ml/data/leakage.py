"""Detect duplicate and near-duplicate images across dataset splits.

A model evaluated on images it has (almost) seen during training reports
optimistic numbers. This module fingerprints every image and reports:

* exact duplicates (identical bytes) that appear in more than one split,
* near-duplicates (difference-hash distance at or below a threshold) across splits,
* label conflicts: the same image filed under different classes,
* groups (e.g. one lesion photographed several times) spread over several splits,
* files that cannot be decoded.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from medvision_ml.data.groups import groups_across_splits
from medvision_ml.data.hashing import dhash, hamming_distances, md5_file
from medvision_ml.data.images import ImageRecord

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fingerprint:
    """Hashes of one decodable image."""

    record: ImageRecord
    md5: str
    dhash: int


@dataclass(frozen=True)
class DuplicatePair:
    """Two images whose fingerprints match closely."""

    first: str
    first_split: str
    first_label: str
    second: str
    second_split: str
    second_label: str
    distance: int

    @property
    def label_conflict(self) -> bool:
        return self.first_label != self.second_label


@dataclass
class LeakageReport:
    """Summary of duplicate and near-duplicate findings."""

    total_images: int
    max_distance: int
    unreadable: list[str] = field(default_factory=list)
    exact_cross_split: list[DuplicatePair] = field(default_factory=list)
    exact_within_split: list[DuplicatePair] = field(default_factory=list)
    near_cross_split: list[DuplicatePair] = field(default_factory=list)
    group_cross_split: dict[str, list[str]] = field(default_factory=dict)

    @property
    def label_conflicts(self) -> list[DuplicatePair]:
        pairs = self.exact_cross_split + self.exact_within_split + self.near_cross_split
        return [p for p in pairs if p.label_conflict]

    @property
    def has_leakage(self) -> bool:
        return bool(self.exact_cross_split or self.near_cross_split or self.group_cross_split)

    def summary(self) -> dict[str, int]:
        return {
            "total_images": self.total_images,
            "unreadable": len(self.unreadable),
            "exact_cross_split_pairs": len(self.exact_cross_split),
            "exact_within_split_pairs": len(self.exact_within_split),
            "near_cross_split_pairs": len(self.near_cross_split),
            "groups_across_splits": len(self.group_cross_split),
            "label_conflicts": len(self.label_conflicts),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "max_distance": self.max_distance,
            "unreadable": self.unreadable,
            "exact_cross_split": [asdict(p) for p in self.exact_cross_split],
            "exact_within_split": [asdict(p) for p in self.exact_within_split],
            "near_cross_split": [asdict(p) for p in self.near_cross_split],
            "group_cross_split": self.group_cross_split,
        }


def fingerprint(record: ImageRecord) -> Fingerprint:
    """Hash one image file.

    Raises:
        OSError: If the file cannot be read or decoded as an image.
    """
    try:
        with Image.open(record.path) as image:
            image.load()
            perceptual = dhash(image)
    except (UnidentifiedImageError, ValueError) as error:
        raise OSError(f"Cannot decode image: {record.path}") from error
    return Fingerprint(record=record, md5=md5_file(record.path), dhash=perceptual)


def fingerprint_all(records: Iterable[ImageRecord]) -> tuple[list[Fingerprint], list[Path]]:
    """Hash every record, returning fingerprints and the paths that failed."""
    fingerprints: list[Fingerprint] = []
    failed: list[Path] = []
    for record in records:
        try:
            fingerprints.append(fingerprint(record))
        except OSError as error:
            LOGGER.warning("%s", error)
            failed.append(record.path)
    return fingerprints, failed


def _pair(a: Fingerprint, b: Fingerprint, distance: int) -> DuplicatePair:
    first, second = sorted((a, b), key=lambda f: (f.record.split, str(f.record.path)))
    return DuplicatePair(
        first=str(first.record.path),
        first_split=first.record.split,
        first_label=first.record.label,
        second=str(second.record.path),
        second_split=second.record.split,
        second_label=second.record.label,
        distance=distance,
    )


def find_exact_duplicates(
    fingerprints: Sequence[Fingerprint],
) -> tuple[list[DuplicatePair], list[DuplicatePair]]:
    """Return exact-duplicate pairs as ``(cross_split, within_split)``."""
    groups: dict[str, list[Fingerprint]] = defaultdict(list)
    for item in fingerprints:
        groups[item.md5].append(item)
    cross: list[DuplicatePair] = []
    within: list[DuplicatePair] = []
    for members in groups.values():
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                target = cross if a.record.split != b.record.split else within
                target.append(_pair(a, b, 0))
    return cross, within


def find_near_duplicates(
    fingerprints: Sequence[Fingerprint],
    max_distance: int,
    *,
    cross_split_only: bool = True,
) -> list[DuplicatePair]:
    """Return image pairs whose dHash distance is at most ``max_distance``.

    Exact byte-for-byte duplicates are excluded; :func:`find_exact_duplicates`
    reports those.
    """
    hashes = np.fromiter((f.dhash for f in fingerprints), dtype=np.uint64, count=len(fingerprints))
    splits = np.array([f.record.split for f in fingerprints])
    pairs: list[DuplicatePair] = []
    for i, item in enumerate(fingerprints[:-1]):
        distances = hamming_distances(item.dhash, hashes[i + 1 :])
        mask = distances <= max_distance
        if cross_split_only:
            mask &= splits[i + 1 :] != item.record.split
        for offset in np.flatnonzero(mask):
            other = fingerprints[i + 1 + int(offset)]
            if other.md5 != item.md5:
                pairs.append(_pair(item, other, int(distances[offset])))
    return pairs


def check_leakage(
    records: Sequence[ImageRecord],
    max_distance: int,
    groups: Mapping[str, str] | None = None,
) -> LeakageReport:
    """Fingerprint ``records`` and report duplicates that cross split boundaries.

    Args:
        records: Images with their split and label.
        max_distance: dHash distance at or below which images count as near-duplicates.
        groups: Optional ``file name -> group`` mapping (e.g. lesion ids); groups
            present in more than one split are reported as leakage.
    """
    LOGGER.info("Fingerprinting %d images.", len(records))
    fingerprints, failed = fingerprint_all(records)
    exact_cross, exact_within = find_exact_duplicates(fingerprints)
    near_cross = find_near_duplicates(fingerprints, max_distance, cross_split_only=True)
    report = LeakageReport(
        total_images=len(records),
        max_distance=max_distance,
        unreadable=[str(p) for p in failed],
        exact_cross_split=exact_cross,
        exact_within_split=exact_within,
        near_cross_split=near_cross,
        group_cross_split=groups_across_splits(records, groups) if groups else {},
    )
    LOGGER.info("Leakage summary: %s", report.summary())
    return report


def group_duplicates(fingerprints: Sequence[Fingerprint], max_distance: int) -> list[int]:
    """Assign a group id to every fingerprint so that near-duplicates share a group.

    Groups are the connected components of the "distance <= max_distance"
    graph, computed with union-find. Splitting by group keeps every copy of an
    image in the same split.
    """
    parent = list(range(len(fingerprints)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    hashes = np.fromiter((f.dhash for f in fingerprints), dtype=np.uint64, count=len(fingerprints))
    for i, item in enumerate(fingerprints[:-1]):
        distances = hamming_distances(item.dhash, hashes[i + 1 :])
        for offset in np.flatnonzero(distances <= max_distance):
            root_a, root_b = find(i), find(i + 1 + int(offset))
            if root_a != root_b:
                parent[root_b] = root_a
    return [find(i) for i in range(len(fingerprints))]
