"""Response models for the v1 API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.inference.decision import Reason, RiskFlag, Status

DISCLAIMER = (
    "Not a medical diagnosis. This tool is for education and screening research only. "
    "Please consult a qualified dermatologist about any skin concern."
)


class ClassProbability(BaseModel):
    class_name: str
    probability: float = Field(ge=0, le=1)


class PredictionResponse(BaseModel):
    status: Status = Field(description="ok, low_confidence or rejected")
    predicted_class: str
    confidence: float = Field(ge=0, le=1, description="Probability of the top class")
    top_k: list[ClassProbability]
    risk_flag: RiskFlag
    malignant_probability: float = Field(
        ge=0, le=1, description="Summed probability of malignant and pre-malignant classes"
    )
    reasons: list[Reason]
    model_version: str
    calibrated: bool = Field(description="Whether probabilities are temperature-calibrated")
    thresholds_fitted: bool = Field(description="Whether decision thresholds were fitted on data")
    summary: str
    disclaimer: str = DISCLAIMER
    latency_ms: float


class ExplanationResponse(PredictionResponse):
    explained_class: str
    heatmap_png: str = Field(description="Base64-encoded PNG of the coloured heatmap")
    overlay_png: str = Field(description="Base64-encoded PNG of the heatmap over the image")


class HealthResponse(BaseModel):
    status: str = "ok"


class ReadyResponse(BaseModel):
    ready: bool
    model_version: str | None = None
    detail: str | None = None


class ThresholdsOut(BaseModel):
    low_confidence: float
    malignant: float
    ood_mahalanobis: float | None = Field(
        default=None,
        description="Squared Mahalanobis distance of the image features above which "
        "the image is rejected as unlike the training data; null when the gate is off",
    )
    max_noise: float | None = Field(
        default=None,
        description="Estimated sensor noise (grey levels) above which the image is "
        "rejected as too noisy; null when the check is off",
    )
    fitted: bool


class MetaResponse(BaseModel):
    model_version: str
    backbone: str
    class_names: list[str]
    malignant_classes: list[str]
    image_size: int
    calibrated: bool
    thresholds: ThresholdsOut
    max_upload_mb: float
    disclaimer: str = DISCLAIMER
