from __future__ import annotations

import json
from pathlib import Path

import pytest
from medvision_ml.cli import main
from medvision_ml.data.groups import (
    group_keys,
    groups_across_splits,
    merge_groups,
    read_groups,
    write_groups,
)
from medvision_ml.data.images import ImageRecord, scan_split_dataset
from medvision_ml.data.isic2019 import GROUPS_FILE, prepare_isic2019, read_labels
from medvision_ml.data.leakage import check_leakage

from tests.conftest import MakeImage


def test_merge_groups_unions_partitions_transitively() -> None:
    # Items 0-1 share a lesion, 1-2 are near-duplicates: all three must merge.
    merged = merge_groups(["a", "a", "b", "c"], [0, 1, 1, 3])
    assert merged[0] == merged[1] == merged[2] != merged[3]
    assert merge_groups() == []
    with pytest.raises(ValueError):
        merge_groups([1, 2], [1])


def test_groups_csv_round_trip_and_validation(tmp_path: Path) -> None:
    write_groups({"b.jpg": "L2", "a.jpg": "L1"}, tmp_path / "g.csv")
    assert read_groups(tmp_path / "g.csv") == {"a.jpg": "L1", "b.jpg": "L2"}
    (tmp_path / "bad.csv").write_text("file,lesion\nx,y\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_groups(tmp_path / "bad.csv")
    (tmp_path / "dup.csv").write_text("image,group\nx,1\nx,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_groups(tmp_path / "dup.csv")


def test_groups_across_splits() -> None:
    records = [
        ImageRecord(Path("a.jpg"), "Train", "MEL"),
        ImageRecord(Path("b.jpg"), "Test", "MEL"),
        ImageRecord(Path("c.jpg"), "Test", "NV"),
    ]
    groups = {"a.jpg": "L1", "b.jpg": "L1"}
    assert group_keys(records, groups) == ["L1", "L1", "image:c.jpg"]
    assert groups_across_splits(records, groups) == {"L1": ["Test", "Train"]}


def test_leakage_report_includes_lesion_groups(tmp_path: Path, make_image: MakeImage) -> None:
    make_image(tmp_path / "Train" / "A" / "x.png", seed=1)
    make_image(tmp_path / "Validation" / "A" / "v.png", seed=3)
    make_image(tmp_path / "Test" / "A" / "y.png", seed=2)  # different pixels, same lesion
    report = check_leakage(scan_split_dataset(tmp_path), 4, {"x.png": "L1", "y.png": "L1"})
    assert report.has_leakage and report.summary()["groups_across_splits"] == 1
    assert not check_leakage(scan_split_dataset(tmp_path), 4).has_leakage


def write_isic(root: Path, make_image: MakeImage) -> tuple[Path, Path, Path]:
    header = "image,MEL,NV,BCC,AK,BKL,DF,VASC,SCC,UNK\n"
    rows = {
        "ISIC_0000001": "1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0",
        "ISIC_0000002": "1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0",
        "ISIC_0000003": "0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0",
    }
    gt = root / "gt.csv"
    gt.write_text(header + "".join(f"{k},{v}\n" for k, v in rows.items()), encoding="utf-8")
    meta = root / "meta.csv"
    meta.write_text(
        "image,age_approx,anatom_site_general,lesion_id,sex\n"
        "ISIC_0000001,50,head/neck,IL_1,male\n"
        "ISIC_0000002,50,head/neck,IL_1,male\n"
        "ISIC_0000003,40,torso,,female\n",
        encoding="utf-8",
    )
    images = root / "images"
    for i, image_id in enumerate(rows):
        make_image(images / f"{image_id}.jpg", seed=i, fmt="JPEG")
    return images, gt, meta


def test_prepare_isic2019(tmp_path: Path, make_image: MakeImage) -> None:
    images, gt, meta = write_isic(tmp_path, make_image)
    counts = prepare_isic2019(images, gt, meta, tmp_path / "out")
    assert counts == {"Melanocytic Nevus": 1, "Melanoma": 2}
    assert (tmp_path / "out" / "Melanoma" / "ISIC_0000001.jpg").is_file()
    assert read_groups(tmp_path / "out" / GROUPS_FILE) == {
        "ISIC_0000001.jpg": "IL_1",
        "ISIC_0000002.jpg": "IL_1",
        "ISIC_0000003.jpg": "image:ISIC_0000003",
    }
    with pytest.raises(FileExistsError):
        prepare_isic2019(images, gt, meta, tmp_path / "out")


def test_prepare_rejects_bad_labels_and_missing_images(
    tmp_path: Path, make_image: MakeImage
) -> None:
    images, gt, meta = write_isic(tmp_path, make_image)
    bad = tmp_path / "bad.csv"
    bad.write_text("image,MEL,NV\nISIC_1,1.0,1.0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_labels(bad)
    (images / "ISIC_0000003.jpg").unlink()
    with pytest.raises(FileNotFoundError):
        prepare_isic2019(images, gt, meta, tmp_path / "out2", link=False)


def test_split_keeps_lesions_together(tmp_path: Path, make_image: MakeImage) -> None:
    (tmp_path / "configs").mkdir()
    config = tmp_path / "configs" / "c.yaml"
    config.write_text("paths:\n  data_dir: data\n  output_dir: out\n", encoding="utf-8")
    raw = tmp_path / "raw"
    groups = {}
    for label, offset in (("A", 0), ("B", 100)):
        for i in range(30):
            name = f"{label}{i}.png"
            make_image(raw / label / name, seed=offset + i)
            groups[name] = f"{label}-lesion-{i // 3}"  # three photos per lesion
    write_groups(groups, raw / GROUPS_FILE)

    assert main(["--config", str(config), "split", "--source", str(raw)]) == 0
    assert (tmp_path / "data" / GROUPS_FILE).is_file()
    assert main(["--config", str(config), "check-leakage"]) == 0
    report = json.loads((tmp_path / "out" / "leakage_report.json").read_text())
    assert report["summary"]["groups_across_splits"] == 0
