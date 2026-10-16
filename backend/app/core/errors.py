"""One place that turns exceptions into ``{"error": {"code", "message"}}`` responses."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.inference.errors import (
    ImageTooLargeError,
    InferenceError,
    ModelArtifactError,
    UnsupportedImageTypeError,
)

LOGGER = logging.getLogger("app.errors")


class APIError(Exception):
    """An error with an HTTP status and a stable code."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def error_body(code: str, message: str) -> dict[str, dict[str, str]]:
    return {"error": {"code": code, "message": message}}


def _status_for(error: InferenceError) -> int:
    if isinstance(error, ImageTooLargeError):
        return 413
    if isinstance(error, UnsupportedImageTypeError):
        return 415
    if isinstance(error, ModelArtifactError):
        return 503
    return 400


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(APIError)
    async def api_error(_: Request, error: APIError) -> JSONResponse:
        return JSONResponse(error_body(error.code, error.message), status_code=error.status_code)

    @app.exception_handler(InferenceError)
    async def inference_error(_: Request, error: InferenceError) -> JSONResponse:
        return JSONResponse(error_body(error.code, error.message), status_code=_status_for(error))

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, error: RequestValidationError) -> JSONResponse:
        details = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in error.errors()
        )
        return JSONResponse(error_body("validation_error", details), status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def http_error(_: Request, error: StarletteHTTPException) -> JSONResponse:
        code = {404: "not_found", 405: "method_not_allowed"}.get(error.status_code, "http_error")
        return JSONResponse(
            error_body(code, str(error.detail)),
            status_code=error.status_code,
            headers=error.headers,
        )

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, error: Exception) -> JSONResponse:
        LOGGER.exception(
            "Unhandled error", extra={"request_id": getattr(request.state, "request_id", None)}
        )
        return JSONResponse(
            error_body("internal_error", "An unexpected error occurred."), status_code=500
        )
