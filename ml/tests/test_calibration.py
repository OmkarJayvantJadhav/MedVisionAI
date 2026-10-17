from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from medvision_ml.calibration.calibrate import read_calibration
from medvision_ml.calibration.fitting import (
    calibrated_probabilities,
    confidence_threshold,
    fit_temperature,
    operating_point,
)


def _overconfident_logits(temperature: float, n: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """Logits whose labels are drawn from softmax(logits / temperature)."""
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 3, size=(n, 5))
    probs = calibrated_probabilities(logits, temperature)
    labels = np.array([rng.choice(5, p=p) for p in probs])
    return logits, labels


def test_fit_temperature_recovers_the_true_temperature() -> None:
    logits, labels = _overconfident_logits(2.5)
    assert fit_temperature(logits, labels) == pytest.approx(2.5, rel=0.1)


def test_temperature_scaling_keeps_the_top_prediction() -> None:
    logits, _ = _overconfident_logits(1.0, n=50)
    for temperature in (0.5, 1.0, 3.0):
        probs = calibrated_probabilities(logits, temperature)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0)
        assert (probs.argmax(axis=1) == logits.argmax(axis=1)).all()


def test_fit_temperature_rejects_mismatched_inputs() -> None:
    with pytest.raises(ValueError):
        fit_temperature(np.zeros((3, 2)), np.zeros(2))


def test_confidence_threshold_keeps_the_most_confident_correct_predictions() -> None:
    probs = np.array([[0.95, 0.05], [0.9, 0.1], [0.8, 0.2], [0.6, 0.4], [0.55, 0.45]])
    labels = np.array([0, 0, 0, 1, 1])  # the two least confident are wrong
    fit = confidence_threshold(probs, labels, target_accuracy=1.0)
    assert fit == {"threshold": 0.8, "coverage": 0.6, "accuracy": 1.0}
    # Out of reach: everything becomes low confidence rather than a guess.
    unreachable = confidence_threshold(probs[3:], labels[3:], target_accuracy=0.9)
    assert unreachable == {"threshold": 1.0, "coverage": 0.0, "accuracy": 0.0}


def test_operating_point_matches_the_backend_rules() -> None:
    # Classes 0 and 1 are malignant; the second image is flagged although class 2 ranks first.
    probs = np.array([[0.7, 0.1, 0.2], [0.15, 0.15, 0.7], [0.0, 0.05, 0.95], [0.5, 0.0, 0.5]])
    labels = np.array([0, 1, 2, 2])
    point = operating_point(probs, labels, [0, 1], malignant_threshold=0.3, confidence=0.6)
    assert point["malignant_recall"] == 1.0
    assert point["malignant_precision"] == pytest.approx(2 / 3)
    assert point["flagged_share"] == 0.75
    assert point["confident_share"] == 0.75
    assert point["confident_accuracy"] == pytest.approx(2 / 3)  # image 2 is confidently wrong


def test_read_calibration_validates_the_file(tmp_path: Path) -> None:
    good = tmp_path / "calibration.json"
    good.write_text(
        json.dumps({"temperature": 1.2, "thresholds": {"low_confidence": 0.6, "malignant": 0.2}})
    )
    assert read_calibration(good) == (1.2, {"low_confidence": 0.6, "malignant": 0.2})
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"temperature": 0, "thresholds": {}}))
    with pytest.raises(ValueError):
        read_calibration(bad)
