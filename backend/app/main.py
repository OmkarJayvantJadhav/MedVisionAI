"""Application factory: ``uvicorn app.main:create_app --factory``."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.formparsers import MultiPartParser

from app import __version__
from app.api.v1.routes import router
from app.core.errors import register_error_handlers
from app.core.logging import configure_logging
from app.core.middleware import RequestContextMiddleware, SlidingWindowLimiter
from app.core.settings import Settings
from app.inference.errors import InferenceError
from app.inference.predictor import Predictor
from app.inference.preprocessing import ImageLimits
from app.services.summary import TemplateSummaryProvider

LOGGER = logging.getLogger("app")
# Room for multipart boundaries and headers on top of the image itself.
MULTIPART_OVERHEAD = 64 * 1024


def _load_predictor(app: FastAPI, settings: Settings) -> None:
    app.state.predictor = None
    app.state.model_error = None
    limits = ImageLimits(max_bytes=settings.max_upload_bytes, max_side=settings.max_image_side)
    try:
        artifacts_predictor = Predictor.from_directory(
            settings.model_dir,
            verify_checksums=settings.verify_checksums,
            threads=settings.inference_threads,
        )
    except InferenceError as error:
        # Start anyway: /health stays up and /ready reports why the model is missing.
        app.state.model_error = error.message
        LOGGER.error(
            "Model not loaded", extra={"model_dir": str(settings.model_dir), "error": error.message}
        )
        return
    artifacts_predictor.limits = limits
    app.state.predictor = artifacts_predictor
    LOGGER.info("Model loaded", extra={"model_version": artifacts_predictor.meta.model_version})


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application. The model is loaded once at startup."""
    settings = settings or Settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        _load_predictor(app, settings)
        yield

    app = FastAPI(
        title="MedVision AI",
        version=__version__,
        summary="Explainable, safety-gated skin-cancer triage for dermoscopy images.",
        description="Not a medical device. Predictions are not diagnoses.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.predictor = None
    app.state.model_error = "Model is loading."
    app.state.summary_provider = TemplateSummaryProvider()

    # Keep uploads in memory: Starlette spools files over 1 MB to a temporary
    # file by default, and uploaded images must never be written to disk.
    MultiPartParser.spool_max_size = max(
        MultiPartParser.spool_max_size, settings.max_upload_bytes + MULTIPART_OVERHEAD
    )

    register_error_handlers(app)
    limiter = (
        SlidingWindowLimiter(settings.rate_limit_per_minute)
        if settings.rate_limit_per_minute
        else None
    )
    app.add_middleware(
        RequestContextMiddleware,
        limiter=limiter,
        trust_forwarded_for=settings.trust_forwarded_for,
        max_body_bytes=settings.max_upload_bytes + MULTIPART_OVERHEAD,
    )
    if settings.allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.allowed_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID"],
        )
    app.include_router(router)
    if settings.static_dir is not None and settings.static_dir.is_dir():
        app.mount("/", StaticFiles(directory=settings.static_dir, html=True), name="frontend")
    return app
