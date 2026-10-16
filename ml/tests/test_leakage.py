from __future__ import annotations

from pathlib import Path

from medvision_ml.data.images import scan_split_dataset
from medvision_ml.data.leakage import check_leakage, fingerprint_all, group_duplicates
from PIL import Image

from tests.conftest import MakeImage


def test_clean_dataset_has_no_leakage(tmp_path: Path, make_image: MakeImage) -> None:
    for i, split in enumerate(("Train", "Validation", "Test")):
        make_image(tmp_path / split / "A" / f"{i}.png", seed=i)
        make_image(tmp_path / split / "B" / f"{i}.png", seed=10 + i)
    report = check_leakage(scan_split_dataset(tmp_path), max_distance=4)
    assert not report.has_leakage
    assert report.summary()["total_images"] == 6


def test_reports_exact_near_within_and_label_conflicts(
    tmp_path: Path, make_image: MakeImage
) -> None:
    # Exact copy across splits.
    src = make_image(tmp_path / "Train" / "A" / "exact.png", seed=1)
    (tmp_path / "Test" / "A").mkdir(parents=True)
    (tmp_path / "Test" / "A" / "exact_copy.png").write_bytes(src.read_bytes())
    # Resized JPEG copy across splits, filed under a different class.
    make_image(tmp_path / "Train" / "A" / "near.png", seed=2, size=128)
    make_image(tmp_path / "Validation" / "B" / "near_copy.jpg", seed=2, size=80, fmt="JPEG")
    # Exact copy inside one split.
    within = make_image(tmp_path / "Train" / "B" / "within.png", seed=3)
    (tmp_path / "Train" / "B" / "within_copy.png").write_bytes(within.read_bytes())
    # Unrelated image and a corrupt file.
    make_image(tmp_path / "Test" / "B" / "unique.png", seed=4)
    (tmp_path / "Test" / "B" / "broken.png").write_bytes(b"not an image")

    report = check_leakage(scan_split_dataset(tmp_path), max_distance=4)

    assert report.has_leakage
    assert [(p.first_split, p.second_split) for p in report.exact_cross_split] == [
        ("Test", "Train")
    ]
    assert len(report.near_cross_split) == 1
    near = report.near_cross_split[0]
    assert {near.first_split, near.second_split} == {"Train", "Validation"}
    assert near.label_conflict
    assert len(report.exact_within_split) == 1
    assert len(report.label_conflicts) == 1
    assert report.unreadable == [str(tmp_path / "Test" / "B" / "broken.png")]
    assert report.to_dict()["summary"]["near_cross_split_pairs"] == 1


def test_group_duplicates_links_transitive_copies(tmp_path: Path, make_image: MakeImage) -> None:
    a = make_image(tmp_path / "Train" / "A" / "a.png", seed=5, size=128)
    Image.open(a).resize((100, 100)).save(tmp_path / "Train" / "A" / "b.png")
    Image.open(a).resize((90, 90)).save(tmp_path / "Train" / "A" / "c.png")
    make_image(tmp_path / "Train" / "A" / "other.png", seed=6)
    for split in ("Validation", "Test"):
        (tmp_path / split).mkdir()
    fingerprints, _ = fingerprint_all(scan_split_dataset(tmp_path))
    groups = dict(
        zip(
            (f.record.path.name for f in fingerprints),
            group_duplicates(fingerprints, 4),
            strict=True,
        )
    )
    assert groups["a.png"] == groups["b.png"] == groups["c.png"] != groups["other.png"]
