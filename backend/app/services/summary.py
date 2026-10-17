"""Plain-language summaries of a prediction.

The template provider only restates the model output; it never adds diagnoses,
treatments or certainty. An optional LLM provider can be added later behind the
same interface (plan Phase 8).
"""

from __future__ import annotations

from typing import Protocol

from app.inference.decision import Reason, RiskFlag, Status
from app.inference.predictor import Prediction

_REJECTION_HINTS = {
    Reason.TOO_DARK: "the photo is too dark",
    Reason.TOO_BRIGHT: "the photo is overexposed",
    Reason.TOO_BLURRY: "the photo is out of focus",
    Reason.TOO_NOISY: "the photo is grainy",
    Reason.OUT_OF_DISTRIBUTION: (
        "it does not look like the dermoscopy images the model was trained on"
    ),
}


class SummaryProvider(Protocol):
    def summarise(self, prediction: Prediction) -> str: ...


class TemplateSummaryProvider:
    """Deterministic summaries built from the prediction fields."""

    def summarise(self, prediction: Prediction) -> str:
        if prediction.status is Status.REJECTED:
            problems = [_REJECTION_HINTS[r] for r in prediction.reasons if r in _REJECTION_HINTS]
            because = " and ".join(problems) or "the image could not be assessed"
            return (
                f"The image could not be assessed reliably because {because}. "
                "Try again with a sharp, well-lit close-up of the lesion."
            )

        top = prediction.top_k[0]
        if prediction.status is Status.LOW_CONFIDENCE:
            others = ", ".join(p.class_name for p in prediction.top_k[1:])
            sentence = (
                f"The model is uncertain. Its best guess is {top.class_name} "
                f"({top.probability:.0%}), followed by {others}."
            )
        else:
            sentence = f"The image most resembles {top.class_name} ({top.probability:.0%})."

        if prediction.risk_flag is RiskFlag.MALIGNANT_SUSPECTED:
            sentence += (
                f" Classes that can be cancerous or pre-cancerous together reach "
                f"{prediction.malignant_probability:.0%}, so a prompt in-person check by a "
                "dermatologist is recommended."
            )
        elif prediction.risk_flag is RiskFlag.UNCERTAIN:
            sentence += " If the lesion changes, bleeds or worries you, see a dermatologist."
        if not prediction.calibrated:
            sentence += " Percentages are uncalibrated model scores, not true probabilities."
        return sentence
