from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from medvision_ml.data.images import ImageRecord, scan_class_dataset, scan_split_dataset
from medvision_ml.data.leakage import check_leakage, fingerprint_all, group_duplicates
from medvision_ml.data.splitting import _allocate, assign_splits, materialise_splits

from tests.conftest import MakeImage

RATIOS = (0.8, 0.1, 0.1)


def records(label: str, n: int) -> list[ImageRecord]:
    return [ImageRecord(Path(f"/src/{label}/{i:03d}.png"), "", label) for i in range(n)]


@pytest.mark.parametrize("count", [0, 1, 7, 10, 35, 877])
def test_allocate_sums_to_count(count: int) -> None:
    sizes = _allocate(count, RATIOS)
    assert sum(sizes) == count
    assert all(abs(s - count * r) < 1 for s, r in zip(sizes, RATIOS, strict=True))


def test_small_classes_still_reach_validation_and_test() -> None:
    assert _allocate(10, RATIOS) == [8, 1, 1]


def test_split_is_stratified_and_deterministic() -> None:
    items = records("A", 100) + records("B", 20)
    groups = list(range(len(items)))
    first = assign_splits(items, groups, RATIOS, seed=42)
    second = assign_splits(items, groups, RATIOS, seed=42)
    assert first == second
    counts = Counter((a.record.label, a.split) for a in first)
    assert counts[("A", "Train")] == 80 and counts[("A", "Test")] == 10
    assert counts[("B", "Train")] == 16 and counts[("B", "Validation")] == 2
    assert assign_splits(items, groups, RATIOS, seed=1) != first


def test_groups_never_straddle_splits() -> None:
    items = records("A", 50)
    groups = [i // 5 for i in range(50)]  # ten groups of five
    split_by_path = {a.record.path: a.split for a in assign_splits(items, groups, RATIOS, seed=0)}
    splits_per_group: dict[int, set[str]] = {}
    for record, group in zip(items, groups, strict=True):
        splits_per_group.setdefault(group, set()).add(split_by_path[record.path])
    assert all(len(splits) == 1 for splits in splits_per_group.values())


def test_uneven_groups_still_fill_every_split() -> None:
    # 12 images: six lesions with two photos each, then six single photos.
    items = records("A", 12)
    groups = [i // 2 if i < 6 else 100 + i for i in range(12)]
    for seed in range(20):
        counts = Counter(a.split for a in assign_splits(items, groups, RATIOS, seed=seed))
        assert counts["Validation"] >= 1 and counts["Test"] >= 1, (seed, counts)
        assert sum(counts.values()) == 12


def test_assign_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        assign_splits(records("A", 3), [0, 1], RATIOS, seed=0)
    with pytest.raises(ValueError):
        assign_splits(records("A", 3), [0, 1, 2], (0.5, 0.5, 0.5), seed=0)


def test_end_to_end_split_has_no_leakage(tmp_path: Path, make_image: MakeImage) -> None:
    source = tmp_path / "raw"
    for label, offset in (("A", 0), ("B", 100)):
        for i in range(20):
            make_image(source / label / f"{i}.png", seed=offset + i)
        # Two resized copies of one image that must stay together.
        make_image(source / label / "dup_a.png", seed=offset + 50, size=128)
        make_image(source / label / "dup_b.png", seed=offset + 50, size=64)
    fingerprints, _ = fingerprint_all(scan_class_dataset(source))
    groups = group_duplicates(fingerprints, 4)
    assignments = assign_splits([f.record for f in fingerprints], groups, RATIOS, seed=42)
    counts = materialise_splits(assignments, tmp_path / "out")
    assert sum(sum(c.values()) for c in counts.values()) == 44
    assert not check_leakage(scan_split_dataset(tmp_path / "out"), 4).has_leakage


def test_materialise_refuses_non_empty_destination(tmp_path: Path, make_image: MakeImage) -> None:
    make_image(tmp_path / "dest" / "existing.png", seed=0)
    with pytest.raises(FileExistsError):
        materialise_splits([], tmp_path / "dest")
