from __future__ import annotations

import json
from pathlib import Path

import pytest
from app.inference.cli import main

from tests.conftest import ImageFactory


def test_cli_prints_predictions_and_saves_overlays(
    model_dir: Path, tmp_path: Path, image_bytes: ImageFactory, capsys: pytest.CaptureFixture[str]
) -> None:
    good = tmp_path / "lesion.png"
    good.write_bytes(image_bytes((40, 230, 40)))
    bad = tmp_path / "notes.txt"
    bad.write_text("hello", encoding="utf-8")
    overlays = tmp_path / "overlays"

    code = main(
        [str(good), str(bad), "--model-dir", str(model_dir), "--overlay-dir", str(overlays)]
    )

    captured = capsys.readouterr()
    assert code == 1  # one file failed
    result = json.loads(captured.out)
    assert result["predicted_class"] == "Melanoma"
    assert result["risk_flag"] == "malignant_suspected"
    assert (overlays / "lesion_overlay.png").is_file()
    assert "notes.txt: error" in captured.err
    assert "Not a medical diagnosis" in captured.err


def test_cli_reports_missing_model(tmp_path: Path) -> None:
    assert main([str(tmp_path / "x.png"), "--model-dir", str(tmp_path / "missing")]) == 2
