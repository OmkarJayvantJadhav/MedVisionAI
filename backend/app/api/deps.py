"""Request-scoped dependencies."""

from __future__ import annotations

from fastapi import Request

from app.core.errors import APIError
from app.core.settings import Settings
from app.inference.predictor import Predictor
from app.services.summary import SummaryProvider


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_predictor(request: Request) -> Predictor:
    predictor: Predictor | None = request.app.state.predictor
    if predictor is None:
        raise APIError(503, "model_unavailable", "The model is not loaded. Try again later.")
    return predictor


def get_summary_provider(request: Request) -> SummaryProvider:
    provider: SummaryProvider = request.app.state.summary_provider
    return provider
