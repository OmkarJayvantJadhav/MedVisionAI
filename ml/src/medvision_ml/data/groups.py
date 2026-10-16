"""Image groups (e.g. several photos of one lesion) that must never be split apart.

A groups file is a CSV with columns ``image`` (file name) and ``group``. Images
missing from the file form their own singleton group.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path

from medvision_ml.data.images import ImageRecord


def read_groups(path: Path) -> dict[str, str]:
    """Read ``image -> group`` from a CSV file.

    Raises:
        ValueError: If the columns are missing or an image appears twice.
    """
    groups: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"image", "group"} <= set(reader.fieldnames):
            raise ValueError(f"{path} must have 'image' and 'group' columns.")
        for row in reader:
            if row["image"] in groups:
                raise ValueError(f"Image listed twice in {path}: {row['image']}")
            groups[row["image"]] = row["group"]
    return groups


def write_groups(groups: Mapping[str, str], path: Path) -> None:
    """Write ``image -> group`` as a CSV file sorted by image name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "group"])
        writer.writerows(sorted(groups.items()))


def group_keys(records: Sequence[ImageRecord], groups: Mapping[str, str]) -> list[str]:
    """Return each record's group, falling back to its own file name."""
    return [groups.get(r.path.name, f"image:{r.path.name}") for r in records]


def merge_groups(*partitions: Sequence[Hashable]) -> list[int]:
    """Combine several groupings of the same items into connected components.

    Two items end up in the same output group if any input partition groups
    them together, directly or through a chain of other items.

    Raises:
        ValueError: If the partitions have different lengths.
    """
    if not partitions:
        return []
    size = len(partitions[0])
    if any(len(p) != size for p in partitions):
        raise ValueError("All partitions must have the same length.")
    parent = list(range(size))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for partition in partitions:
        first_seen: dict[Hashable, int] = {}
        for index, key in enumerate(partition):
            if key in first_seen:
                a, b = find(first_seen[key]), find(index)
                if a != b:
                    parent[b] = a
            else:
                first_seen[key] = index
    return [find(i) for i in range(size)]


def groups_across_splits(
    records: Sequence[ImageRecord], groups: Mapping[str, str]
) -> dict[str, list[str]]:
    """Return the groups whose images appear in more than one split, with those splits."""
    splits: dict[str, set[str]] = defaultdict(set)
    for record, key in zip(records, group_keys(records, groups), strict=True):
        splits[key].add(record.split)
    return {key: sorted(found) for key, found in sorted(splits.items()) if len(found) > 1}
