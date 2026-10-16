"""Deterministic, stratified, duplicate-aware train/validation/test splitting."""

from __future__ import annotations

import hashlib
import random
import shutil
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from medvision_ml.data.images import SPLITS, ImageRecord


@dataclass(frozen=True)
class SplitAssignment:
    """The split chosen for one source image."""

    record: ImageRecord
    split: str


def _stable_seed(seed: int, label: str) -> int:
    """Derive a per-class seed that does not depend on Python's hash randomisation."""
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _allocate(count: int, ratios: Sequence[float]) -> list[int]:
    """Split ``count`` items into integer sizes that follow ``ratios`` and sum to ``count``.

    Uses the largest-remainder method so small classes still receive validation
    and test items whenever their share rounds to at least one.
    """
    exact = [count * r for r in ratios]
    sizes = [int(x) for x in exact]
    order = sorted(range(len(ratios)), key=lambda i: exact[i] - sizes[i], reverse=True)
    for i in order[: count - sum(sizes)]:
        sizes[i] += 1
    return sizes


def assign_splits(
    records: Sequence[ImageRecord],
    groups: Sequence[int],
    ratios: Sequence[float],
    seed: int,
    split_names: Sequence[str] = SPLITS,
) -> list[SplitAssignment]:
    """Assign every record to a split, stratified by label and grouped by duplicates.

    All records that share a group id land in the same split, so near-duplicate
    images can never leak from training into evaluation. Groups are shuffled per
    class with a stable seed, then allocated to splits by cumulative image count.

    Args:
        records: Source images (their ``split`` field is ignored).
        groups: Group id per record, from :func:`~medvision_ml.data.leakage.group_duplicates`.
        ratios: Target fraction per split; must sum to 1.
        seed: Base random seed.
        split_names: Output split names, aligned with ``ratios``.

    Raises:
        ValueError: If the inputs are inconsistent.
    """
    if len(records) != len(groups):
        raise ValueError("records and groups must have the same length.")
    if len(ratios) != len(split_names):
        raise ValueError("ratios and split_names must have the same length.")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("ratios must sum to 1.")

    by_label: dict[str, dict[int, list[ImageRecord]]] = defaultdict(lambda: defaultdict(list))
    for record, group in zip(records, groups, strict=True):
        by_label[record.label][group].append(record)

    assignments: list[SplitAssignment] = []
    for label in sorted(by_label):
        label_groups = [sorted(g, key=lambda r: str(r.path)) for g in by_label[label].values()]
        label_groups.sort(key=lambda g: str(g[0].path))
        random.Random(_stable_seed(seed, label)).shuffle(label_groups)

        total = sum(len(g) for g in label_groups)
        targets = _allocate(total, ratios)
        boundaries = [sum(targets[: i + 1]) for i in range(len(targets))]
        assigned = 0
        for group_records in label_groups:
            # Place the whole group in the split whose quota its first image falls into.
            split_index = next(i for i, b in enumerate(boundaries) if assigned < b)
            assignments.extend(
                SplitAssignment(record=r, split=split_names[split_index]) for r in group_records
            )
            assigned += len(group_records)
    return assignments


def materialise_splits(
    assignments: Sequence[SplitAssignment],
    destination: Path,
    *,
    overwrite: bool = False,
) -> dict[str, dict[str, int]]:
    """Copy assigned images into ``destination/<split>/<label>/`` and return counts.

    Raises:
        FileExistsError: If ``destination`` is not empty and ``overwrite`` is false.
    """
    if destination.exists() and any(destination.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Destination is not empty: {destination}")
        shutil.rmtree(destination)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for item in assignments:
        target_dir = destination / item.split / item.record.label
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / item.record.path.name
        if target.exists():
            raise FileExistsError(f"Two source images map to the same file name: {target}")
        shutil.copy2(item.record.path, target)
        counts[item.split][item.record.label] += 1
    return {split: dict(labels) for split, labels in counts.items()}
