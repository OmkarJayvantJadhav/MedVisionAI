from __future__ import annotations

import math

import pytest
from medvision_ml.training.class_weights import compute_class_weights

COUNTS = {0: 900, 1: 90, 2: 10}


def sample_weighted_mean(weights: dict[int, float]) -> float:
    return sum(weights[k] * COUNTS[k] for k in COUNTS) / sum(COUNTS.values())


def test_none_returns_no_weights() -> None:
    assert compute_class_weights(COUNTS, "none") is None


@pytest.mark.parametrize("scheme", ["balanced", "sqrt"])
def test_weights_are_normalised_and_favour_rare_classes(scheme: str) -> None:
    weights = compute_class_weights(COUNTS, scheme)  # type: ignore[arg-type]
    assert weights is not None
    assert math.isclose(sample_weighted_mean(weights), 1.0)
    assert weights[0] < weights[1] < weights[2]


def test_sqrt_is_gentler_than_balanced() -> None:
    balanced = compute_class_weights(COUNTS, "balanced")
    sqrt = compute_class_weights(COUNTS, "sqrt")
    assert balanced is not None and sqrt is not None
    assert sqrt[2] / sqrt[0] == pytest.approx(math.sqrt(balanced[2] / balanced[0]))


def test_empty_class_is_rejected() -> None:
    with pytest.raises(ValueError):
        compute_class_weights({0: 5, 1: 0}, "balanced")
