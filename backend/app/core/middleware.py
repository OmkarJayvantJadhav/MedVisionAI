"""Request IDs, access logging and a per-client rate limit."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.errors import error_body

LOGGER = logging.getLogger("app.access")
RATE_LIMITED_PREFIX = "/api/"


def client_ip(request: Request, trust_forwarded_for: bool) -> str:
    """Return the client address, optionally from the first ``X-Forwarded-For`` hop."""
    if trust_forwarded_for:
        forwarded = request.headers.get("x-forwarded-for", "")
        first = forwarded.split(",")[0].strip()
        if first:
            return first
    return request.client.host if request.client else "unknown"


class SlidingWindowLimiter:
    """Allow at most ``limit`` requests per client in any ``window`` seconds.

    State is in memory, which suits the single-instance deployment; several
    instances would need a shared store.
    """

    def __init__(
        self, limit: int, window: float = 60.0, clock: Callable[[], float] = time.monotonic
    ):
        self.limit = limit
        self.window = window
        self.clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> float | None:
        """Record a request; return seconds to wait if over the limit, else ``None``."""
        now = self.clock()
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] <= now - self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return max(0.0, hits[0] + self.window - now)
            hits.append(now)
            if len(self._hits) > 10_000:  # drop idle clients so memory stays bounded
                for stale in [
                    k for k, v in self._hits.items() if not v or v[-1] <= now - self.window
                ]:
                    del self._hits[stale]
            return None


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request ID, enforce rate and body-size limits, and log one line per request.

    Bodies whose ``Content-Length`` exceeds ``max_body_bytes`` are refused before
    they are read.
    """

    def __init__(
        self,
        app: ASGIApp,
        limiter: SlidingWindowLimiter | None,
        trust_forwarded_for: bool,
        max_body_bytes: int,
    ):
        super().__init__(app)
        self.limiter = limiter
        self.trust_forwarded_for = trust_forwarded_for
        self.max_body_bytes = max_body_bytes

    def _body_too_large(self, request: Request) -> bool:
        length = request.headers.get("content-length", "")
        return length.isdigit() and int(length) > self.max_body_bytes

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request_id = request_id[:64]
        request.state.request_id = request_id
        started = time.perf_counter()

        response: Response
        retry_after = None
        if self.limiter and request.url.path.startswith(RATE_LIMITED_PREFIX):
            retry_after = self.limiter.check(client_ip(request, self.trust_forwarded_for))
        if retry_after is not None:
            response = JSONResponse(
                error_body("rate_limited", "Too many requests; please slow down."),
                status_code=429,
                headers={"Retry-After": str(max(1, round(retry_after)))},
            )
        elif self._body_too_large(request):
            response = JSONResponse(
                error_body("image_too_large", "The request body exceeds the upload limit."),
                status_code=413,
            )
        else:
            response = await call_next(request)

        response.headers["X-Request-ID"] = request_id
        LOGGER.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            },
        )
        return response
