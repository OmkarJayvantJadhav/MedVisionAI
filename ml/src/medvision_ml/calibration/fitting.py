"""Fit the temperature and the decision thresholds shipped with a model.

Everything here is fitted on the validation split only; the test split is used
to report how the fitted values hold up on unseen data.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Search range for log(T): temperatures from about 0.05 to 20.
_LOG_T_BOUNDS = (-3.0, 3.0)
_GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0


def _log_softmax(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = scores - scores.max(axis=1, keepdims=True)
    result: NDArray[np.float64] = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return result


def calibrated_probabilities(logits: Any, temperature: float) -> NDArray[np.float64]:
    """Return ``softmax(logits / temperature)`` row by row."""
    if temperature <= 0:
        raise ValueError("temperature must be positive.")
    scores = np.asarray(logits, dtype=np.float64)
    return np.exp(_log_softmax(scores / temperature))


def fit_temperature(logits: Any, labels: Any) -> float:
    """Return the temperature that minimises the negative log-likelihood of ``labels``.

    Temperature scaling divides every logit by one positive number, so it changes
    confidence but never the ranking of classes or the top-1 prediction.

    Raises:
        ValueError: If the inputs are empty or their shapes do not match.
    """
    scores = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[0] != targets.shape[0]:
        raise ValueError("logits must be (n, classes) with one label per row.")
    rows = np.arange(targets.shape[0])

    def nll(log_t: float) -> float:
        return float(-_log_softmax(scores / np.exp(log_t))[rows, targets].mean())

    # Golden-section search on log(T); the loss has a single minimum there.
    low, high = _LOG_T_BOUNDS
    while high - low > 1e-6:
        left = high - _GOLDEN * (high - low)
        right = low + _GOLDEN * (high - low)
        if nll(left) <= nll(right):
            high = right
        else:
            low = left
    return float(np.exp((low + high) / 2))


def confidence_threshold(
    probabilities: Any, labels: Any, target_accuracy: float
) -> dict[str, float]:
    """Find the lowest top-1 confidence above which predictions reach ``target_accuracy``.

    Images below the threshold are reported as ``low_confidence``. When no
    threshold reaches the target, the threshold is 1.0, so every answer is
    reported as low confidence and the report shows zero coverage.

    Returns:
        The threshold, the share of images at or above it (coverage) and their
        accuracy (0 when nothing is covered).
    """
    probs = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == targets
    order = np.argsort(-confidence)
    # Accuracy of the k most confident predictions, for every k.
    running = np.cumsum(correct[order]) / np.arange(1, order.size + 1)
    reaching = np.flatnonzero(running >= target_accuracy)
    threshold = float(confidence[order][reaching.max()]) if reaching.size else 1.0
    accepted = confidence >= threshold
    return {
        "threshold": threshold,
        "coverage": float(accepted.mean()),
        "accuracy": float(correct[accepted].mean()) if accepted.any() else 0.0,
    }


def operating_point(
    probabilities: Any,
    labels: Any,
    malignant: Sequence[int],
    malignant_threshold: float,
    confidence: float,
) -> dict[str, float]:
    """Score fitted thresholds on a split, as the backend would apply them."""
    probs = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    group = np.asarray(malignant)
    is_malignant = np.isin(targets, group)
    flagged = probs[:, group].sum(axis=1) >= malignant_threshold
    accepted = probs.max(axis=1) >= confidence
    correct = probs.argmax(axis=1) == targets
    return {
        "malignant_recall": float(flagged[is_malignant].mean()) if is_malignant.any() else 0.0,
        "malignant_precision": float(is_malignant[flagged].mean()) if flagged.any() else 0.0,
        "flagged_share": float(flagged.mean()),
        "confident_share": float(accepted.mean()),
        "confident_accuracy": float(correct[accepted].mean()) if accepted.any() else 0.0,
    }
