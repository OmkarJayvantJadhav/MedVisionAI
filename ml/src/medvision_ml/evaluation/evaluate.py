"""Evaluate a saved model on one dataset split and write a report folder."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from medvision_ml.config import ProjectConfig
from medvision_ml.data.dataset import load_split
from medvision_ml.evaluation.metrics import classification_metrics, reliability_bins
from medvision_ml.evaluation.plots import plot_confusion_matrix, plot_reliability_diagram
from medvision_ml.models.io import load_class_names, load_model

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Predictions:
    """Model outputs for every sample of a split, in file order."""

    file_paths: list[str]
    labels: NDArray[np.int64]
    probabilities: NDArray[np.float64]


def predict_split(
    model: Any, split_dir: Path, config: ProjectConfig
) -> tuple[Predictions, list[str]]:
    """Run ``model`` over an unshuffled split and collect labels and probabilities."""
    split = load_split(split_dir, config.data, shuffle=False, seed=config.seed)
    labels: list[NDArray[np.int64]] = []
    probabilities: list[NDArray[np.float64]] = []
    for images, batch_labels in split.dataset:
        probabilities.append(np.asarray(model(images, training=False), dtype=np.float64))
        labels.append(np.asarray(batch_labels, dtype=np.int64))
    predictions = Predictions(
        file_paths=split.file_paths,
        labels=np.concatenate(labels),
        probabilities=np.concatenate(probabilities),
    )
    return predictions, split.class_names


def _report_text(metrics: dict[str, Any]) -> str:
    lines = [f"{'Class':<28}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Support':>10}"]
    for name, row in metrics["per_class"].items():
        lines.append(
            f"{name:<28}{row['precision']:>10.4f}{row['recall']:>10.4f}"
            f"{row['f1']:>10.4f}{row['support']:>10d}"
        )
    lines.append("")
    for key in ("accuracy", "macro_f1", "weighted_f1", "roc_auc_ovr_macro", "ece"):
        lines.append(f"{key:<28}{metrics[key]:.4f}")
    group = metrics.get("malignant_group")
    if group:
        lines.append(f"{'malignant_recall_top1':<28}{group['recall_top1']:.4f}")
        lines.append(f"{'malignant_precision_top1':<28}{group['precision_top1']:.4f}")
        lines.append(f"{'malignant_auroc':<28}{group['auroc']:.4f}")
    return "\n".join(lines) + "\n"


def write_predictions_csv(predictions: Predictions, class_names: list[str], path: Path) -> None:
    """Write one row per sample with the true class, prediction and all probabilities."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "true", "predicted", "confidence", *class_names])
        for file, label, probs in zip(
            predictions.file_paths, predictions.labels, predictions.probabilities, strict=True
        ):
            top = int(probs.argmax())
            writer.writerow(
                [file, class_names[label], class_names[top], f"{probs[top]:.6f}"]
                + [f"{p:.6f}" for p in probs]
            )


def evaluate(
    config: ProjectConfig,
    model_path: Path,
    split: str = "Test",
    output_dir: Path | None = None,
    *,
    trust_legacy_lambda: bool = False,
) -> dict[str, Any]:
    """Evaluate ``model_path`` on ``config.data_dir/<split>`` and write a report.

    The report folder contains ``metrics.json``, ``classification_report.txt``,
    ``predictions.csv``, ``confusion_matrix.png`` and ``reliability.png``.

    Raises:
        ValueError: If the dataset classes differ from the model's class names.
    """
    names_path = model_path.with_name(f"{model_path.stem}_class_names.json")
    model_classes = load_class_names(names_path)
    model = load_model(model_path, trust_legacy_lambda=trust_legacy_lambda)
    predictions, class_names = predict_split(model, config.data_dir / split, config)
    if class_names != model_classes:
        raise ValueError(f"Dataset classes {class_names} differ from model classes {model_classes}")

    metrics = classification_metrics(
        predictions.labels,
        predictions.probabilities,
        class_names,
        config.evaluation.malignant_classes,
        config.evaluation.calibration_bins,
    )
    metrics["model"] = str(model_path)
    metrics["split"] = split

    out = output_dir or config.output_dir / "evaluation" / model_path.parent.name / split.lower()
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "classification_report.txt").write_text(_report_text(metrics), encoding="utf-8")
    write_predictions_csv(predictions, class_names, out / "predictions.csv")
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, out / "confusion_matrix.png")
    bins = reliability_bins(
        predictions.labels, predictions.probabilities, config.evaluation.calibration_bins
    )
    plot_reliability_diagram(bins, metrics["ece"], out / "reliability.png")
    LOGGER.info(
        "Evaluation on %s: accuracy %.4f, macro F1 %.4f, report in %s",
        split,
        metrics["accuracy"],
        metrics["macro_f1"],
        out,
    )
    return metrics
