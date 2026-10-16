from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from medvision_ml.cli import main
from medvision_ml.data.dedupe import deduplicate
from medvision_ml.data.images import ImageRecord, scan_class_dataset
from medvision_ml.data.leakage import fingerprint_all
from medvision_ml.data.splitting import assign_splits
from PIL import Image

from tests.conftest import MakeImage


def test_deduplicate_keeps_one_copy_and_drops_conflicts(
    tmp_path: Path, make_image: MakeImage
) -> None:
    same = make_image(tmp_path / "A" / "same.png", seed=1)
    (tmp_path / "A" / "same_copy.png").write_bytes(same.read_bytes())
    clash = make_image(tmp_path / "A" / "clash.png", seed=2)
    (tmp_path / "B").mkdir()
    (tmp_path / "B" / "clash_copy.png").write_bytes(clash.read_bytes())
    make_image(tmp_path / "B" / "unique.png", seed=3)

    fingerprints, _ = fingerprint_all(scan_class_dataset(tmp_path))
    kept, report = deduplicate(fingerprints)

    assert sorted(f.record.path.name for f in kept) == ["same.png", "unique.png"]
    assert report.summary() == {
        "duplicate_copies_removed": 1,
        "conflicting_label_images_removed": 2,
    }


def test_mixed_label_group_lands_in_one_split() -> None:
    records = [ImageRecord(Path(f"/x/A{i}.png"), "", "A") for i in range(20)]
    records += [ImageRecord(Path(f"/x/B{i}.png"), "", "B") for i in range(20)]
    groups = list(range(40))
    groups[39] = groups[0]  # B19 is a near-copy of A0 with a different label
    for seed in range(10):
        by_path = {
            a.record.path.name: a.split
            for a in assign_splits(records, groups, (0.8, 0.1, 0.1), seed)
        }
        assert by_path["A0.png"] == by_path["B19.png"]
        assert len(by_path) == 40


def test_split_has_no_leakage_with_conflicting_duplicates(
    tmp_path: Path, make_image: MakeImage
) -> None:
    (tmp_path / "configs").mkdir()
    config = tmp_path / "configs" / "c.yaml"
    config.write_text("paths:\n  data_dir: data\n  output_dir: out\n", encoding="utf-8")
    raw = tmp_path / "raw"
    for label, offset in (("A", 0), ("B", 100)):
        for i in range(20):
            make_image(raw / label / f"{label}{i}.png", seed=offset + i, size=128)
    # Same file filed under both classes, a resized copy under the other class,
    # and a plain duplicate with the same label.
    (raw / "B" / "A0_copy.png").write_bytes((raw / "A" / "A0.png").read_bytes())
    Image.open(raw / "A" / "A1.png").resize((90, 90)).save(raw / "B" / "A1_small.png")
    (raw / "A" / "A2_dup.png").write_bytes((raw / "A" / "A2.png").read_bytes())

    assert main(["--config", str(config), "split", "--source", str(raw)]) == 0
    assert main(["--config", str(config), "check-leakage"]) == 0
    removed = json.loads((tmp_path / "out" / "split_removed.json").read_text())
    assert len(removed["label_conflicts"]) == 2 and len(removed["duplicate_copies"]) == 1
    files = Counter(p.name for p in (tmp_path / "data").rglob("*.png"))
    assert "A0.png" not in files and "A0_copy.png" not in files and "A2_dup.png" not in files
