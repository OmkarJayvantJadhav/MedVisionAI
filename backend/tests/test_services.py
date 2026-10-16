from __future__ import annotations

import logging

import pytest
from app.core.logging import JsonFormatter
from app.core.middleware import SlidingWindowLimiter
from app.inference.decision import Reason, RiskFlag, Status
from app.inference.predictor import ClassProbability, Prediction
from app.services.summary import TemplateSummaryProvider


def prediction(**overrides: object) -> Prediction:
    fields: dict[str, object] = {
        "status": Status.OK,
        "predicted_class": "Nevus",
        "confidence": 0.82,
        "top_k": (
            ClassProbability("Nevus", 0.82),
            ClassProbability("Melanoma", 0.1),
            ClassProbability("Acne", 0.08),
        ),
        "risk_flag": RiskFlag.BENIGN,
        "malignant_probability": 0.1,
        "reasons": (),
        "model_version": "v",
        "calibrated": True,
        "thresholds_fitted": True,
        "latency_ms": 1.0,
    }
    fields.update(overrides)
    return Prediction(**fields)  # type: ignore[arg-type]


SUMMARY = TemplateSummaryProvider()


def test_confident_benign_summary() -> None:
    text = SUMMARY.summarise(prediction())
    assert text == "The image most resembles Nevus (82%)."


def test_malignant_and_uncalibrated_summary() -> None:
    text = SUMMARY.summarise(
        prediction(
            risk_flag=RiskFlag.MALIGNANT_SUSPECTED, malignant_probability=0.4, calibrated=False
        )
    )
    assert "40%" in text and "dermatologist" in text and "uncalibrated" in text


def test_low_confidence_lists_alternatives() -> None:
    text = SUMMARY.summarise(prediction(status=Status.LOW_CONFIDENCE, risk_flag=RiskFlag.UNCERTAIN))
    assert "uncertain" in text and "Melanoma, Acne" in text and "dermatologist" in text


@pytest.mark.parametrize(
    ("reasons", "phrase"),
    [
        ((Reason.TOO_DARK, Reason.TOO_BLURRY), "too dark and the photo is out of focus"),
        ((Reason.OUT_OF_DISTRIBUTION,), "does not look like"),
        ((), "could not be assessed"),
    ],
)
def test_rejected_summary_explains_why(reasons: tuple[Reason, ...], phrase: str) -> None:
    text = SUMMARY.summarise(
        prediction(status=Status.REJECTED, risk_flag=RiskFlag.UNCERTAIN, reasons=reasons)
    )
    assert phrase in text and "Try again" in text


def test_summaries_never_claim_a_diagnosis() -> None:
    for status in Status:
        for risk in RiskFlag:
            text = SUMMARY.summarise(prediction(status=status, risk_flag=risk)).lower()
            assert "you have" not in text and "diagnosed" not in text


def test_sliding_window_limiter() -> None:
    now = [0.0]
    limiter = SlidingWindowLimiter(limit=2, window=60, clock=lambda: now[0])
    assert limiter.check("a") is None and limiter.check("a") is None
    assert limiter.check("a") == pytest.approx(60)
    assert limiter.check("b") is None  # other clients are independent
    now[0] = 30
    assert limiter.check("a") == pytest.approx(30)
    now[0] = 60.5
    assert limiter.check("a") is None


def test_json_formatter_includes_extras_and_exceptions() -> None:
    record = logging.makeLogRecord(
        {"msg": "hello %s", "args": ("world",), "levelname": "INFO", "request_id": "r1"}
    )
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record.exc_info = sys.exc_info()
    text = JsonFormatter().format(record)
    assert '"message": "hello world"' in text and '"request_id": "r1"' in text and "boom" in text
