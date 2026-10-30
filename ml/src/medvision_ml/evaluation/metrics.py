"""Classification, safety and calibration metrics computed from probabilities.

Everything here is pure NumPy / scikit-learn so it can be tested without a model.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn import metrics as skm

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ReliabilityBins:
    """Per-bin statistics for a reliability diagram."""

    edges: FloatArray
    confidence: FloatArray
    accuracy: FloatArray
    counts: IntArray


def validate_inputs(y_true: Any, probabilities: Any) -> tuple[IntArray, FloatArray]:
    """Return labels and probabilities as checked NumPy arrays.

    Raises:
        ValueError: On shape mismatches, non-finite values or out-of-range labels.
    """
    labels = np.asarray(y_true, dtype=np.int64).reshape(-1)
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] < 2:
        raise ValueError("probabilities must have shape (n_samples, n_classes >= 2).")
    if probs.shape[0] != labels.shape[0] or labels.size == 0:
        raise ValueError("y_true and probabilities must have the same non-zero length.")
    if not np.all(np.isfinite(probs)):
        raise ValueError("probabilities must be finite.")
    if labels.min() < 0 or labels.max() >= probs.shape[1]:
        raise ValueError("y_true contains a label without a probability column.")
    return labels, probs


def reliability_bins(y_true: Any, probabilities: Any, n_bins: int = 15) -> ReliabilityBins:
    """Bin samples by top-1 confidence and compare confidence with accuracy."""
    labels, probs = validate_inputs(y_true, probabilities)
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Right-closed bins, with confidence 0 placed in the first bin.
    index = np.clip(np.searchsorted(edges, confidence, side="left") - 1, 0, n_bins - 1)
    counts = np.bincount(index, minlength=n_bins).astype(np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_conf = np.bincount(index, weights=confidence, minlength=n_bins) / counts
        mean_acc = np.bincount(index, weights=correct, minlength=n_bins) / counts
    return ReliabilityBins(edges=edges, confidence=mean_conf, accuracy=mean_acc, counts=counts)


def expected_calibration_error(y_true: Any, probabilities: Any, n_bins: int = 15) -> float:
    """Top-label expected calibration error with equal-width bins."""
    bins = reliability_bins(y_true, probabilities, n_bins)
    filled = bins.counts > 0
    gaps = np.abs(bins.confidence[filled] - bins.accuracy[filled])
    return float(np.sum(gaps * bins.counts[filled]) / bins.counts.sum())


def malignant_indices(class_names: Sequence[str], malignant: Sequence[str]) -> list[int]:
    """Map malignant class names to indices.

    Raises:
        ValueError: If a malignant class is not among ``class_names``.
    """
    missing = sorted(set(malignant) - set(class_names))
    if missing:
        raise ValueError(f"Malignant classes not found in class names: {missing}")
    return sorted(class_names.index(name) for name in malignant)


def malignant_group_metrics(
    y_true: Any,
    probabilities: Any,
    group: Sequence[int],
) -> dict[str, float]:
    """Treat the malignant classes as one positive group and score the model on it.

    ``recall_top1`` is the share of truly malignant images whose top-1 prediction
    is any malignant class; ``auroc`` ranks images by their summed malignant
    probability.
    """
    labels, probs = validate_inputs(y_true, probabilities)
    group_arr = np.asarray(group)
    is_positive = np.isin(labels, group_arr)
    predicted_positive = np.isin(probs.argmax(axis=1), group_arr)
    score = probs[:, group_arr].sum(axis=1)
    result = {
        "recall_top1": float(skm.recall_score(is_positive, predicted_positive, zero_division=0)),
        "precision_top1": float(
            skm.precision_score(is_positive, predicted_positive, zero_division=0)
        ),
        "auroc": float("nan"),
    }
    if 0 < is_positive.sum() < is_positive.size:
        result["auroc"] = float(skm.roc_auc_score(is_positive, score))
    return result


def threshold_for_recall(
    y_true: Any,
    probabilities: Any,
    group: Sequence[int],
    target_recall: float,
) -> dict[str, float]:
    """Find the highest summed-malignant-probability threshold reaching ``target_recall``.

    Returns:
        The threshold and the recall and precision it achieves on these samples.

    Raises:
        ValueError: If no malignant samples are present.
    """
    labels, probs = validate_inputs(y_true, probabilities)
    group_arr = np.asarray(group)
    is_positive = np.isin(labels, group_arr)
    if not is_positive.any():
        raise ValueError("No malignant samples to fit a threshold on.")
    score = probs[:, group_arr].sum(axis=1)
    # A threshold at the k-th highest positive score recalls at least k positives.
    positives = np.sort(score[is_positive])[::-1]
    needed = int(np.ceil(target_recall * positives.size))
    threshold = float(positives[max(needed, 1) - 1])
    flagged = score >= threshold
    return {
        "threshold": threshold,
        "recall": float(flagged[is_positive].mean()),
        "precision": float(is_positive[flagged].mean()),
    }


def classification_metrics(
    y_true: Any,
    probabilities: Any,
    class_names: Sequence[str],
    malignant: Sequence[str] = (),
    n_bins: int = 15,
) -> dict[str, Any]:
    """Return the full metric bundle used for reports and model selection."""
    labels, probs = validate_inputs(y_true, probabilities)
    if len(class_names) != probs.shape[1]:
        raise ValueError("class_names must have one entry per probability column.")
    predicted = probs.argmax(axis=1)
    all_labels = np.arange(probs.shape[1])

    report = skm.classification_report(
        labels,
        predicted,
        labels=all_labels,
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    present = np.unique(labels)
    auc = float("nan")
    if present.size == probs.shape[1] == 2:
        auc = float(skm.roc_auc_score(labels, probs[:, 1]))
    elif present.size == probs.shape[1]:
        auc = float(skm.roc_auc_score(labels, probs, multi_class="ovr", average="macro"))

    result: dict[str, Any] = {
        "n_samples": int(labels.size),
        "accuracy": float(skm.accuracy_score(labels, predicted)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "roc_auc_ovr_macro": auc,
        "ece": expected_calibration_error(labels, probs, n_bins),
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in class_names
        },
        "confusion_matrix": skm.confusion_matrix(labels, predicted, labels=all_labels).tolist(),
    }
    if malignant:
        group = malignant_indices(class_names, malignant)
        result["malignant_group"] = malignant_group_metrics(labels, probs, group)
    return result
