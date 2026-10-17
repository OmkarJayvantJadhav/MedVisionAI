"""Version 1 endpoints."""

from __future__ import annotations

import base64
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.api.deps import get_predictor, get_settings, get_summary_provider
from app.core.errors import APIError
from app.core.settings import Settings
from app.inference.errors import ImageTooLargeError
from app.inference.predictor import Prediction, Predictor
from app.schemas.prediction import (
    ClassProbability,
    ExplanationResponse,
    HealthResponse,
    MetaResponse,
    PredictionResponse,
    ReadyResponse,
    ThresholdsOut,
)
from app.services.summary import SummaryProvider

router = APIRouter(prefix="/api/v1")

ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"description": "Invalid image"},
    413: {"description": "Image too large"},
    415: {"description": "Unsupported image type"},
    422: {"description": "Invalid request"},
    429: {"description": "Rate limited"},
    503: {"description": "Model not loaded"},
}

PredictorDep = Annotated[Predictor, Depends(get_predictor)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
SummaryDep = Annotated[SummaryProvider, Depends(get_summary_provider)]
ImageFile = Annotated[UploadFile, File(description="JPEG, PNG or WebP image")]


async def read_upload(upload: UploadFile, settings: Settings) -> bytes:
    """Read an upload, refusing to buffer more than the configured limit."""
    data = await upload.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise ImageTooLargeError(f"Image exceeds {settings.max_upload_mb:g} MB.")
    return data


def _base_fields(prediction: Prediction, summary: str) -> dict[str, Any]:
    return {
        "status": prediction.status,
        "predicted_class": prediction.predicted_class,
        "confidence": prediction.confidence,
        "top_k": [
            ClassProbability(class_name=p.class_name, probability=p.probability)
            for p in prediction.top_k
        ],
        "risk_flag": prediction.risk_flag,
        "malignant_probability": prediction.malignant_probability,
        "reasons": list(prediction.reasons),
        "model_version": prediction.model_version,
        "calibrated": prediction.calibrated,
        "thresholds_fitted": prediction.thresholds_fitted,
        "summary": summary,
        "latency_ms": round(prediction.latency_ms, 1),
    }


@router.get("/health", response_model=HealthResponse, tags=["service"])
def health() -> HealthResponse:
    """Liveness: the process is up."""
    return HealthResponse()


@router.get(
    "/ready",
    response_model=ReadyResponse,
    tags=["service"],
    responses={503: {"model": ReadyResponse}},
)
def ready(request: Request) -> Any:
    """Readiness: the model is loaded and its checksums were verified."""
    predictor: Predictor | None = request.app.state.predictor
    if predictor is None:
        body = ReadyResponse(ready=False, detail=request.app.state.model_error)
        return JSONResponse(body.model_dump(), status_code=503)
    return ReadyResponse(ready=True, model_version=predictor.meta.model_version)


@router.get(
    "/meta", response_model=MetaResponse, tags=["service"], responses={503: ERROR_RESPONSES[503]}
)
def meta(predictor: PredictorDep, settings: SettingsDep) -> MetaResponse:
    """Model version, classes, thresholds and upload limits."""
    m = predictor.meta
    return MetaResponse(
        model_version=m.model_version,
        backbone=m.backbone,
        class_names=list(m.class_names),
        malignant_classes=list(m.malignant_classes),
        image_size=m.image_size,
        calibrated=m.calibration_fitted,
        thresholds=ThresholdsOut(
            low_confidence=m.thresholds.low_confidence,
            malignant=m.thresholds.malignant,
            ood_mahalanobis=m.thresholds.ood_mahalanobis,
            max_noise=m.thresholds.max_noise,
            fitted=m.thresholds.fitted,
        ),
        max_upload_mb=settings.max_upload_mb,
    )


@router.post(
    "/predict", response_model=PredictionResponse, tags=["inference"], responses=ERROR_RESPONSES
)
async def predict(
    file: ImageFile, predictor: PredictorDep, settings: SettingsDep, summaries: SummaryDep
) -> PredictionResponse:
    """Classify a skin-lesion image."""
    data = await read_upload(file, settings)
    prediction = await run_in_threadpool(predictor.predict, data)
    return PredictionResponse(**_base_fields(prediction, summaries.summarise(prediction)))


@router.post(
    "/explain", response_model=ExplanationResponse, tags=["inference"], responses=ERROR_RESPONSES
)
async def explain(
    file: ImageFile,
    predictor: PredictorDep,
    settings: SettingsDep,
    summaries: SummaryDep,
    class_name: Annotated[
        str | None, Query(description="Class to explain (default: top prediction)")
    ] = None,
    alpha: Annotated[float, Query(ge=0, le=1, description="Heatmap opacity in the overlay")] = 0.4,
) -> ExplanationResponse:
    """Classify an image and return a Grad-CAM heatmap and overlay."""
    if class_name is not None and class_name not in predictor.meta.class_names:
        raise APIError(400, "unknown_class", f"Unknown class '{class_name}'.")
    data = await read_upload(file, settings)
    prediction = await run_in_threadpool(
        lambda: predictor.predict(data, explain=True, explain_class=class_name, overlay_alpha=alpha)
    )
    explanation = prediction.explanation
    if explanation is None:  # pragma: no cover - predict(explain=True) always sets it
        raise APIError(500, "internal_error", "Explanation was not produced.")
    return ExplanationResponse(
        **_base_fields(prediction, summaries.summarise(prediction)),
        explained_class=explanation.class_name,
        heatmap_png=base64.b64encode(explanation.heatmap_png).decode("ascii"),
        overlay_png=base64.b64encode(explanation.overlay_png).decode("ascii"),
    )
