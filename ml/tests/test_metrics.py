from __future__ import annotations

import math

import numpy as np
import pytest
from medvision_ml.evaluation.metrics import (
    classification_metrics,
    expected_calibration_error,
    malignant_group_metrics,
    malignant_indices,
    reliability_bins,
    selection_key,
    threshold_for_recall,
)

NAMES = ["Acne", "Melanoma", "Nevus"]


def one_hot(labels: list[int], n: int = 3, confidence: float = 1.0) -> np.ndarray:
    probs = np.full((len(labels), n), (1 - confidence) / (n - 1))
    probs[np.arange(len(labels)), labels] = confidence
    return probs


def test_perfect_predictions() -> None:
    y = [0, 1, 2, 1]
    metrics = classification_metrics(y, one_hot(y), NAMES, malignant=["Melanoma"])
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["roc_auc_ovr_macro"] == 1.0
    assert metrics["ece"] == pytest.approx(0.0)
    assert metrics["malignant_group"]["recall_top1"] == 1.0
    assert metrics["per_class"]["Melanoma"]["support"] == 2
    assert metrics["confusion_matrix"] == [[1, 0, 0], [0, 2, 0], [0, 0, 1]]


def test_binary_problem_uses_positive_class_score() -> None:
    probs = np.array([[0.9, 0.1], [0.4, 0.6], [0.7, 0.3], [0.2, 0.8]])
    metrics = classification_metrics([0, 1, 1, 1], probs, ["Nevus", "Melanoma"])
    assert metrics["roc_auc_ovr_macro"] == pytest.approx(1.0)
    assert metrics["accuracy"] == 0.75


def test_auc_is_nan_when_a_class_is_missing() -> None:
    y = [0, 0, 2]
    assert math.isnan(classification_metrics(y, one_hot(y), NAMES)["roc_auc_ovr_macro"])


def test_ece_of_overconfident_model() -> None:
    # Always 90% confident but right only half the time: ECE = 0.4.
    y = [0, 1, 0, 1]
    probs = one_hot([0, 0, 0, 0], confidence=0.9)
    assert expected_calibration_error(y, probs, n_bins=10) == pytest.approx(0.4)
    bins = reliability_bins(y, probs, n_bins=10)
    assert bins.counts.sum() == 4 and bins.counts[8] == 4


def test_malignant_group_counts_any_malignant_class_as_a_hit() -> None:
    names = ["Acne", "BCC", "Melanoma", "Nevus"]
    group = malignant_indices(names, ["BCC", "Melanoma"])
    y = [2, 2, 1, 3]
    preds = one_hot([1, 3, 1, 3], n=4, confidence=0.7)  # melanoma predicted as BCC counts
    result = malignant_group_metrics(y, preds, group)
    assert result["recall_top1"] == pytest.approx(2 / 3)
    assert result["precision_top1"] == 1.0


def test_unknown_malignant_class_is_rejected() -> None:
    with pytest.raises(ValueError):
        malignant_indices(NAMES, ["Carcinoma"])


def test_threshold_for_recall_reaches_target() -> None:
    y = np.array([1, 1, 1, 1, 0, 2])
    probs = np.array(
        [
            [0.1, 0.8, 0.1],
            [0.3, 0.4, 0.3],
            [0.5, 0.2, 0.3],
            [0.6, 0.1, 0.3],
            [0.2, 0.3, 0.5],
            [0.9, 0.05, 0.05],
        ]
    )
    result = threshold_for_recall(y, probs, [1], target_recall=0.75)
    assert result["threshold"] == pytest.approx(0.2)
    assert result["recall"] == 0.75
    assert result["precision"] == pytest.approx(3 / 4)


def test_threshold_requires_positives() -> None:
    with pytest.raises(ValueError):
        threshold_for_recall([0, 2], one_hot([0, 2]), [1], 0.9)


def test_selection_prefers_malignant_recall_then_f1_then_calibration() -> None:
    a = {"malignant_group": {"recall_top1": 0.9}, "macro_f1": 0.6, "ece": 0.10}
    b = {"malignant_group": {"recall_top1": 0.8}, "macro_f1": 0.9, "ece": 0.01}
    c = {"malignant_group": {"recall_top1": 0.9}, "macro_f1": 0.6, "ece": 0.05}
    assert max([a, b, c], key=selection_key) is c


@pytest.mark.parametrize(
    ("labels", "probs"),
    [
        ([0, 1], np.ones((3, 3)) / 3),
        ([0, 5], np.ones((2, 3)) / 3),
        ([0], np.array([[np.nan, 0.5, 0.5]])),
        ([0], np.ones((1, 1))),
    ],
)
def test_invalid_inputs_raise(labels: list[int], probs: np.ndarray) -> None:
    with pytest.raises(ValueError):
        classification_metrics(
            labels, probs, NAMES[: probs.shape[1]] if probs.shape[1] <= 3 else NAMES
        )
