"""Calibration, quality, CAM and decision rules."""

from __future__ import annotations

import numpy as np
import pytest
from app.inference.artifacts import Thresholds
from app.inference.calibration import (
    energy_score,
    logits_from_features,
    mahalanobis_distance,
    pooled_features,
    softmax,
)
from app.inference.cam import blend, cam_from_features, encode_png, jet, render_heatmap
from app.inference.decision import Reason, RiskFlag, Status, decide
from app.inference.quality import QualityReport, assess, laplacian_variance, noise_level

from tests.conftest import textured

THRESHOLDS = Thresholds(
    low_confidence=0.5,
    malignant=0.3,
    ood_energy=None,
    min_brightness=20.0,
    max_brightness=235.0,
    min_sharpness=5.0,
    fitted=False,
)
GOOD = QualityReport(brightness=120.0, sharpness=50.0)
MALIGNANT = (1,)


def test_logits_are_pooled_features_times_kernel() -> None:
    features = np.arange(2 * 2 * 2 * 3, dtype=np.float32).reshape(2, 2, 2, 3)
    kernel = np.eye(3, dtype=np.float32)
    bias = np.array([1.0, 0.0, -1.0], dtype=np.float32)
    expected = features.mean(axis=(1, 2)) + bias
    np.testing.assert_allclose(logits_from_features(features, kernel, bias), expected)


def test_softmax_temperature_flattens_and_stays_normalised() -> None:
    logits = np.array([[4.0, 1.0, 0.0]])
    sharp, flat = softmax(logits), softmax(logits, temperature=3.0)
    assert sharp.sum() == pytest.approx(1.0) and flat.sum() == pytest.approx(1.0)
    assert flat.max() < sharp.max()
    np.testing.assert_allclose(softmax(np.array([[1000.0, 0.0]])), [[1.0, 0.0]])


def test_energy_is_higher_for_less_confident_logits() -> None:
    confident, unsure = energy_score(np.array([[10.0, 0.0, 0.0], [0.1, 0.0, 0.0]]))
    assert unsure > confident
    assert energy_score(np.array([[0.0, 0.0]]))[0] == pytest.approx(-np.log(2))


def test_quality_measures() -> None:
    flat = np.full((32, 32, 3), 100.0)
    assert laplacian_variance(flat[..., 0]) == 0.0
    report = assess(textured((100, 100, 100), size=(32, 32)).astype(np.float32))
    assert 90 < report.brightness < 110 and report.sharpness > 5.0
    assert assess(flat).noise == 0.0 and report.noise > 5.0


def test_noise_level_ignores_local_texture() -> None:
    y, x = np.mgrid[0:128, 0:128]
    gray = (x + y) * 1.0
    assert noise_level(gray) == pytest.approx(0.0, abs=1e-9)
    gray[:40, :40] += np.random.default_rng(1).normal(0, 40, (40, 40))
    assert noise_level(gray) < 0.5
    noisy = (x + y) + np.random.default_rng(2).normal(0, 10, (128, 128))
    assert noise_level(noisy) == pytest.approx(7.3, rel=0.1)
    assert noise_level(np.zeros((9, 9))) == 0.0


def test_noise_gate_only_applies_once_fitted() -> None:
    noisy = QualityReport(brightness=120.0, sharpness=50.0, noise=6.0)
    confident = probs(0.9, 0.05, 0.05)
    assert decide(confident, MALIGNANT, THRESHOLDS, noisy, -5.0).status is Status.OK
    gated = Thresholds(**{**vars(THRESHOLDS), "max_noise": 2.5})
    decision = decide(confident, MALIGNANT, gated, noisy, -5.0)
    assert decision.status is Status.REJECTED and decision.reasons == (Reason.TOO_NOISY,)
    assert decide(confident, MALIGNANT, gated, GOOD, -5.0).status is Status.OK


def test_cam_follows_kernel_column_and_is_normalised() -> None:
    features = np.zeros((2, 2, 2), dtype=np.float32)
    features[0, 0, 0] = 4.0
    features[1, 1, 1] = 2.0
    kernel = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float32)
    np.testing.assert_allclose(cam_from_features(features, kernel, 0), [[1.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(cam_from_features(features, kernel, 1), np.zeros((2, 2)))


def test_jet_endpoints_and_rendering() -> None:
    colours = jet(np.array([0.0, 0.5, 1.0]))
    assert colours[0].tolist() == [0, 0, 128]
    assert colours[1].tolist() == [128, 255, 128]
    assert colours[2].tolist() == [128, 0, 0]
    assert render_heatmap(np.eye(4, dtype=np.float32), 20, 30).shape == (20, 30, 3)


def test_blend_and_png() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    colour = np.full((4, 4, 3), 200, dtype=np.uint8)
    assert blend(image, colour, 0.5)[0, 0].tolist() == [100, 100, 100]
    with pytest.raises(ValueError):
        blend(image, colour, 1.5)
    assert encode_png(image).startswith(b"\x89PNG")


def probs(*values: float) -> np.ndarray:
    return np.array(values)


def test_confident_benign_is_ok() -> None:
    decision = decide(probs(0.9, 0.05, 0.05), MALIGNANT, THRESHOLDS, GOOD, energy=-5.0)
    assert decision.status is Status.OK and decision.risk_flag is RiskFlag.BENIGN
    assert decision.reasons == ()


def test_malignant_flag_overrides_benign_top_class() -> None:
    decision = decide(probs(0.6, 0.35, 0.05), MALIGNANT, THRESHOLDS, GOOD, energy=-5.0)
    assert decision.status is Status.OK
    assert decision.risk_flag is RiskFlag.MALIGNANT_SUSPECTED
    assert decision.malignant_probability == pytest.approx(0.35)


def test_low_confidence() -> None:
    decision = decide(probs(0.4, 0.2, 0.4), MALIGNANT, THRESHOLDS, GOOD, energy=-1.0)
    assert decision.status is Status.LOW_CONFIDENCE and decision.risk_flag is RiskFlag.UNCERTAIN
    assert decision.reasons == (Reason.LOW_CONFIDENCE,)


def test_low_confidence_can_still_flag_malignancy() -> None:
    decision = decide(probs(0.3, 0.45, 0.25), MALIGNANT, THRESHOLDS, GOOD, energy=-1.0)
    assert decision.status is Status.LOW_CONFIDENCE
    assert decision.risk_flag is RiskFlag.MALIGNANT_SUSPECTED


@pytest.mark.parametrize(
    ("quality", "reason"),
    [
        (QualityReport(5.0, 50.0), Reason.TOO_DARK),
        (QualityReport(250.0, 50.0), Reason.TOO_BRIGHT),
        (QualityReport(120.0, 1.0), Reason.TOO_BLURRY),
    ],
)
def test_unusable_images_are_rejected(quality: QualityReport, reason: Reason) -> None:
    decision = decide(probs(0.1, 0.8, 0.1), MALIGNANT, THRESHOLDS, quality, energy=-5.0)
    assert decision.status is Status.REJECTED and decision.risk_flag is RiskFlag.UNCERTAIN
    assert decision.reasons == (reason,)


def test_ood_gate_only_applies_once_fitted() -> None:
    fitted = Thresholds(**{**vars(THRESHOLDS), "ood_energy": -3.0})
    assert decide(probs(0.9, 0.05, 0.05), MALIGNANT, THRESHOLDS, GOOD, -1.0).status is Status.OK
    decision = decide(probs(0.9, 0.05, 0.05), MALIGNANT, fitted, GOOD, energy=-1.0)
    assert decision.status is Status.REJECTED
    assert decision.reasons == (Reason.OUT_OF_DISTRIBUTION,)


def test_mahalanobis_distance_of_pooled_features() -> None:
    features = np.zeros((2, 2, 2, 3), dtype=np.float32)
    features[1] = 1.0
    pooled = pooled_features(features)
    np.testing.assert_array_equal(pooled, [[0, 0, 0], [1, 1, 1]])
    precision = np.diag([1.0, 4.0, 0.25])
    np.testing.assert_allclose(mahalanobis_distance(pooled, np.zeros(3), precision), [0, 5.25])


def test_feature_gate_rejects_distant_images() -> None:
    gated = Thresholds(**{**vars(THRESHOLDS), "ood_mahalanobis": 100.0})
    confident = probs(0.9, 0.05, 0.05)
    assert decide(confident, MALIGNANT, gated, GOOD, -5.0, distance=99.0).status is Status.OK
    decision = decide(confident, MALIGNANT, gated, GOOD, -5.0, distance=101.0)
    assert decision.status is Status.REJECTED
    assert decision.reasons == (Reason.OUT_OF_DISTRIBUTION,)
    # Without a fitted threshold the distance is ignored.
    assert decide(confident, MALIGNANT, THRESHOLDS, GOOD, -5.0, distance=1e9).status is Status.OK
