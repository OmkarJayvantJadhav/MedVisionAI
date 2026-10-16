"""End-to-end smoke test: train, evaluate and explain on a tiny synthetic dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from medvision_ml.cli import main

from tests.conftest import MakeImage

CONFIG = """
seed: 1
paths:
  data_dir: data
  output_dir: out
  model_dir: artifacts
data:
  image_size: 32
  batch_size: 4
  cache_dir: cache
model:
  pretrained: false
training:
  epochs: 1
  fine_tune_epochs: 1
  fine_tune_layers: 5
  class_weighting: sqrt
evaluation:
  malignant_classes: [Melanoma]
"""


@pytest.mark.slow
def test_train_evaluate_and_explain(tmp_path: Path, make_image: MakeImage) -> None:
    (tmp_path / "configs").mkdir()
    config = tmp_path / "configs" / "smoke.yaml"
    config.write_text(CONFIG, encoding="utf-8")
    for split, count in (("Train", 6), ("Validation", 2), ("Test", 2)):
        for label, offset in (("Melanoma", 0), ("Nevus", 500)):
            for i in range(count):
                make_image(
                    tmp_path / "data" / split / label / f"{i}.png",
                    seed=offset + hash(split) % 97 + i,
                )

    args = ["--config", str(config)]
    assert main([*args, "train", "--run-name", "smoke"]) == 0
    model = tmp_path / "artifacts" / "smoke" / "model.keras"
    assert model.is_file()
    run_dir = tmp_path / "out" / "runs" / "smoke"
    assert json.loads((run_dir / "training_summary.json").read_text())["epochs_run"] == 2
    assert (run_dir / "history.png").is_file()
    assert any((tmp_path / "cache").glob("train*")), "training split should be cached on disk"

    assert main([*args, "evaluate", "--model", str(model)]) == 0
    report = tmp_path / "out" / "evaluation" / "smoke" / "test"
    metrics = json.loads((report / "metrics.json").read_text())
    assert metrics["n_samples"] == 4
    assert set(metrics["malignant_group"]) == {"recall_top1", "precision_top1", "auroc"}
    for name in (
        "classification_report.txt",
        "predictions.csv",
        "confusion_matrix.png",
        "reliability.png",
    ):
        assert (report / name).is_file()

    image = tmp_path / "data" / "Test" / "Melanoma" / "0.png"
    overlay = tmp_path / "overlay.png"
    assert (
        main(
            [
                *args,
                "gradcam",
                "--model",
                str(model),
                "--image",
                str(image),
                "--output",
                str(overlay),
            ]
        )
        == 0
    )
    assert overlay.is_file()
