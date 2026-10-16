"""Turning model outputs into a safety status and a risk flag.

The rules, in order:

1. ``rejected``: the image is too dark, too bright or too blurry, or (once an
   OOD threshold is fitted) does not look like the training data.
2. ``low_confidence``: the top calibrated probability is below the threshold.
3. ``ok`` otherwise.

``risk_flag`` is ``malignant_suspected`` whenever the summed probability of the
malignant and pre-malignant classes reaches its threshold, even if a benign class
ranks first. Rejected images are always ``uncertain``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

from app.inference.artifacts import Thresholds
from app.inference.quality import QualityReport


class Status(StrEnum):
    OK = "ok"
    LOW_CONFIDENCE = "low_confidence"
    REJECTED = "rejected"


class RiskFlag(StrEnum):
    MALIGNANT_SUSPECTED = "malignant_suspected"
    BENIGN = "benign"
    UNCERTAIN = "uncertain"


class Reason(StrEnum):
    TOO_DARK = "too_dark"
    TOO_BRIGHT = "too_bright"
    TOO_BLURRY = "too_blurry"
    OUT_OF_DISTRIBUTION = "out_of_distribution"
    LOW_CONFIDENCE = "low_confidence"
    MALIGNANT_PROBABILITY = "malignant_probability_above_threshold"


@dataclass(frozen=True)
class Decision:
    status: Status
    risk_flag: RiskFlag
    malignant_probability: float
    reasons: tuple[Reason, ...]


def decide(
    probabilities: NDArray[np.floating],
    malignant_indices: tuple[int, ...],
    thresholds: Thresholds,
    quality: QualityReport,
    energy: float,
) -> Decision:
    """Apply the safety rules to one image's calibrated probabilities."""
    reasons: list[Reason] = []
    if quality.brightness < thresholds.min_brightness:
        reasons.append(Reason.TOO_DARK)
    elif quality.brightness > thresholds.max_brightness:
        reasons.append(Reason.TOO_BRIGHT)
    if quality.sharpness < thresholds.min_sharpness:
        reasons.append(Reason.TOO_BLURRY)
    if thresholds.ood_energy is not None and energy > thresholds.ood_energy:
        reasons.append(Reason.OUT_OF_DISTRIBUTION)

    malignant = float(np.sum(probabilities[list(malignant_indices)])) if malignant_indices else 0.0
    if reasons:
        return Decision(Status.REJECTED, RiskFlag.UNCERTAIN, malignant, tuple(reasons))

    status = Status.OK
    if float(np.max(probabilities)) < thresholds.low_confidence:
        status = Status.LOW_CONFIDENCE
        reasons.append(Reason.LOW_CONFIDENCE)

    if malignant_indices and malignant >= thresholds.malignant:
        risk = RiskFlag.MALIGNANT_SUSPECTED
        reasons.append(Reason.MALIGNANT_PROBABILITY)
    elif status is Status.LOW_CONFIDENCE:
        risk = RiskFlag.UNCERTAIN
    else:
        risk = RiskFlag.BENIGN
    return Decision(status, risk, malignant, tuple(reasons))
