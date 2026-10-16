from __future__ import annotations

import json
from pathlib import Path

import pytest
from medvision_ml.cli import main

from tests.conftest import MakeImage


def make_config(tmp_path: Path) -> Path:
    configs = tmp_path / "configs"
    configs.mkdir()
    path = configs / "test.yaml"
    path.write_text("paths:\n  data_dir: data\n  output_dir: out\n", encoding="utf-8")
    return path


def test_split_then_check_leakage(
    tmp_path: Path, make_image: MakeImage, capsys: pytest.CaptureFixture[str]
) -> None:
    config = make_config(tmp_path)
    for label, offset in (("A", 0), ("B", 50)):
        for i in range(10):
            make_image(tmp_path / "raw" / label / f"{i}.png", seed=offset + i)

    assert main(["--config", str(config), "split", "--source", str(tmp_path / "raw")]) == 0
    assert (tmp_path / "data" / "Train" / "A").is_dir()

    assert main(["--config", str(config), "check-leakage"]) == 0
    report = json.loads((tmp_path / "out" / "leakage_report.json").read_text(encoding="utf-8"))
    assert report["summary"]["total_images"] == 20

    # Copy a training image into the test split: the check must now fail.
    leaked = next((tmp_path / "data" / "Train" / "A").iterdir())
    (tmp_path / "data" / "Test" / "A" / "leaked.png").write_bytes(leaked.read_bytes())
    assert main(["--config", str(config), "check-leakage"]) == 1
    assert "Leakage found" in capsys.readouterr().err


def test_errors_return_exit_code_2(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    assert (
        main(["--config", str(config), "check-leakage", "--data-dir", str(tmp_path / "missing")])
        == 2
    )
