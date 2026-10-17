from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from medvision_ml.calibration.ood import (
    OOD_STATS_FILE,
    InputGates,
    acceptance_threshold,
    auroc,
    energy,
    fit_gaussian,
    mahalanobis,
    read_ood,
)


def test_mahalanobis_matches_the_definition() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(500, 4)) @ rng.normal(size=(4, 4))
    mean, precision = fit_gaussian(features, shrinkage=0.0)
    np.testing.assert_allclose(mean, features.mean(axis=0))
    np.testing.assert_allclose(
        precision, np.linalg.inv(np.cov(features.T, bias=True)), rtol=1e-6, atol=1e-8
    )
    point = features[:3]
    expected = [(p - mean) @ precision @ (p - mean) for p in point]
    np.testing.assert_allclose(mahalanobis(point, mean, precision), expected)
    # Under a fitted Gaussian the mean squared distance equals the dimension.
    assert mahalanobis(features, mean, precision).mean() == pytest.approx(4.0)


def test_shrinkage_keeps_few_samples_invertible() -> None:
    features = np.random.default_rng(1).normal(size=(5, 50))
    mean, precision = fit_gaussian(features, shrinkage=0.1)
    assert np.isfinite(precision).all()
    far = mean + 100.0
    assert mahalanobis(far[None], mean, precision)[0] > mahalanobis(features, mean, precision).max()


@pytest.mark.parametrize(
    ("features", "shrinkage"),
    [(np.zeros((1, 3)), 0.0), (np.zeros(3), 0.0), (np.zeros((4, 3)), 2.0)],
)
def test_fit_gaussian_rejects_bad_input(features: np.ndarray, shrinkage: float) -> None:
    with pytest.raises(ValueError):
        fit_gaussian(features, shrinkage)


def test_energy_is_negative_logsumexp() -> None:
    logits = np.array([[1.0, 2.0, 3.0], [1000.0, 0.0, -5.0]])
    expected = -2.0 * np.log(np.exp(logits[0] / 2.0).sum())
    result = energy(logits, temperature=2.0)
    assert result[0] == pytest.approx(expected)
    assert result[1] == pytest.approx(-1000.0)


def test_auroc() -> None:
    assert auroc([1, 2, 3], [4, 5]) == 1.0
    assert auroc([4, 5], [1, 2, 3]) == 0.0
    assert auroc([1, 2], [1, 2]) == 0.5
    with pytest.raises(ValueError):
        auroc([], [1.0])


def test_acceptance_threshold_lets_the_target_share_through() -> None:
    scores = np.arange(1, 1001, dtype=float)
    threshold = acceptance_threshold(scores, 0.99)
    assert threshold == 990.0
    # Never fewer than the target, even when the quantile falls between scores.
    for n in (7, 101, 2528):
        values = np.random.default_rng(n).normal(size=n)
        assert np.mean(values <= acceptance_threshold(values, 0.99)) >= 0.99
        assert acceptance_threshold(values, 0.99) in values
    with pytest.raises(ValueError):
        acceptance_threshold(scores, 0.0)
    with pytest.raises(ValueError):
        acceptance_threshold([], 0.99)


def test_read_ood(tmp_path: Path) -> None:
    report = tmp_path / "ood.json"
    report.write_text(json.dumps({"threshold": 1800.5}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        read_ood(report)
    np.savez(tmp_path / OOD_STATS_FILE, mean=np.zeros(2), precision=np.eye(2))
    assert read_ood(report) == InputGates(1800.5, tmp_path / OOD_STATS_FILE, None)
    report.write_text(json.dumps({"threshold": 1800.5, "noise": {"max_noise": 2.4}}))
    assert read_ood(report).max_noise == 2.4
    report.write_text(json.dumps({"threshold": -1}), encoding="utf-8")
    with pytest.raises(ValueError, match="threshold"):
        read_ood(report)
