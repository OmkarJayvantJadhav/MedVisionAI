"""Fit calibration and thresholds for a saved model and write a report folder."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import keras
import numpy as np
from numpy.typing import NDArray

from medvision_ml.calibration.fitting import (
    calibrated_probabilities,
    confidence_threshold,
    fit_temperature,
    operating_point,
)
from medvision_ml.config import ProjectConfig
from medvision_ml.data.dataset import load_split
from medvision_ml.evaluation.metrics import (
    expected_calibration_error,
    malignant_indices,
    reliability_bins,
    threshold_for_recall,
)
from medvision_ml.evaluation.plots import plot_reliability_diagram
from medvision_ml.models.builder import LOGITS_LAYER
from medvision_ml.models.io import load_class_names, load_model

LOGGER = logging.getLogger(__name__)

CALIBRATION_FILE = "calibration.json"


def predict_logits(
    model: keras.Model, split_dir: Path, config: ProjectConfig
) -> tuple[NDArray[np.float64], NDArray[np.int64], list[str]]:
    """Return the pre-softmax logits and labels of an unshuffled split, and its classes."""
    logits_model = keras.Model(model.input, model.get_layer(LOGITS_LAYER).output)
    split = load_split(split_dir, config.data, shuffle=False, seed=config.seed)
    logits: list[NDArray[np.float64]] = []
    labels: list[NDArray[np.int64]] = []
    for images, batch_labels in split.dataset:
        logits.append(np.asarray(logits_model(images, training=False), dtype=np.float64))
        labels.append(np.asarray(batch_labels, dtype=np.int64))
    return np.concatenate(logits), np.concatenate(labels), split.class_names


def calibrate(
    config: ProjectConfig,
    model_path: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Fit the temperature and thresholds on Validation and report them on Test.

    Writes ``calibration.json`` (read by ``medvision-ml export --calibration``)
    and reliability diagrams before and after temperature scaling.

    Raises:
        ValueError: If the dataset classes differ from the model's class names.
    """
    targets = config.evaluation
    model_classes = load_class_names(model_path.with_name(f"{model_path.stem}_class_names.json"))
    model = load_model(model_path)
    malignant = malignant_indices(model_classes, targets.malignant_classes)
    bins = targets.calibration_bins
    out = output_dir or config.output_dir / "calibration" / model_path.parent.name
    out.mkdir(parents=True, exist_ok=True)

    splits: dict[str, tuple[NDArray[np.float64], NDArray[np.int64]]] = {}
    for split in ("Validation", "Test"):
        logits, labels, class_names = predict_logits(model, config.data_dir / split, config)
        if class_names != model_classes:
            raise ValueError(f"Dataset classes {class_names} differ from {model_classes}")
        splits[split] = (logits, labels)

    val_logits, val_labels = splits["Validation"]
    temperature = fit_temperature(val_logits, val_labels)
    val_probs = calibrated_probabilities(val_logits, temperature)
    malignant_fit = threshold_for_recall(
        val_labels, val_probs, malignant, targets.target_malignant_recall
    )
    confidence_fit = confidence_threshold(val_probs, val_labels, targets.target_confident_accuracy)

    report: dict[str, Any] = {
        "model": str(model_path),
        "temperature": temperature,
        "thresholds": {
            "low_confidence": confidence_fit["threshold"],
            "malignant": malignant_fit["threshold"],
        },
        "targets": {
            "malignant_recall": targets.target_malignant_recall,
            "confident_accuracy": targets.target_confident_accuracy,
        },
        "splits": {},
    }
    for split, (logits, labels) in splits.items():
        raw = calibrated_probabilities(logits, 1.0)
        scaled = calibrated_probabilities(logits, temperature)
        report["splits"][split.lower()] = {
            "n_samples": int(labels.size),
            "ece_before": expected_calibration_error(labels, raw, bins),
            "ece_after": expected_calibration_error(labels, scaled, bins),
            **operating_point(
                scaled,
                labels,
                malignant,
                malignant_fit["threshold"],
                confidence_fit["threshold"],
            ),
        }
        for name, probs in (("before", raw), ("after", scaled)):
            ece = report["splits"][split.lower()][f"ece_{name}"]
            plot_reliability_diagram(
                reliability_bins(labels, probs, bins),
                ece,
                out / f"reliability_{split.lower()}_{name}.png",
            )

    (out / CALIBRATION_FILE).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    LOGGER.info(
        "Calibration: T=%.3f, malignant threshold %.3f, low-confidence threshold %.3f; %s",
        temperature,
        malignant_fit["threshold"],
        confidence_fit["threshold"],
        out,
    )
    return report


def read_calibration(path: Path) -> tuple[float, dict[str, float]]:
    """Read the temperature and fitted thresholds from a ``calibration.json`` file.

    Raises:
        ValueError: If the file lacks a positive temperature or either threshold.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    temperature = data.get("temperature")
    thresholds = data.get("thresholds", {})
    if not isinstance(temperature, int | float) or temperature <= 0:
        raise ValueError(f"{path} has no positive temperature.")
    if not {"low_confidence", "malignant"} <= set(thresholds):
        raise ValueError(f"{path} must contain low_confidence and malignant thresholds.")
    return float(temperature), {k: float(thresholds[k]) for k in ("low_confidence", "malignant")}
