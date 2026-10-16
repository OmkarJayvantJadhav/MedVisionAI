from __future__ import annotations

import base64
import io
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from app.core.logging import JsonFormatter
from app.core.settings import Settings
from app.main import create_app
from app.schemas.prediction import DISCLAIMER
from fastapi.testclient import TestClient
from PIL import Image

from tests.conftest import CLASSES, ImageFactory, encode, textured


def make_client(**overrides: Any) -> TestClient:
    settings = Settings(_env_file=None, **{"rate_limit_per_minute": 0, **overrides})  # type: ignore[call-arg]
    return TestClient(create_app(settings))


@pytest.fixture
def client(model_dir: Path) -> Iterator[TestClient]:
    with make_client(model_dir=model_dir) as test_client:
        yield test_client


def upload(data: bytes, name: str = "lesion.png") -> dict[str, Any]:
    return {"file": (name, data, "application/octet-stream")}


def assert_error(response: Any, status: int, code: str) -> None:
    assert response.status_code == status, response.text
    body = response.json()
    assert set(body) == {"error"} and body["error"]["code"] == code
    assert body["error"]["message"]


def test_health_ready_and_meta(client: TestClient) -> None:
    assert client.get("/api/v1/health").json() == {"status": "ok"}
    assert client.get("/api/v1/ready").json() == {
        "ready": True,
        "model_version": "test-v0.0.1",
        "detail": None,
    }
    meta = client.get("/api/v1/meta").json()
    assert meta["class_names"] == CLASSES
    assert meta["malignant_classes"] == ["Melanoma"]
    assert meta["thresholds"]["fitted"] is False and meta["calibrated"] is False
    assert meta["disclaimer"] == DISCLAIMER


def test_predict_flags_malignancy(client: TestClient, image_bytes: ImageFactory) -> None:
    response = client.post("/api/v1/predict", files=upload(image_bytes((40, 230, 40))))
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["predicted_class"] == "Melanoma"
    assert body["risk_flag"] == "malignant_suspected"
    assert body["reasons"] == ["malignant_probability_above_threshold"]
    assert len(body["top_k"]) == 3 and body["top_k"][0]["class_name"] == "Melanoma"
    assert "dermatologist" in body["summary"] and "uncalibrated" in body["summary"]
    assert body["disclaimer"] == DISCLAIMER
    assert body["model_version"] == "test-v0.0.1" and body["latency_ms"] >= 0
    assert response.headers["x-request-id"]


def test_predict_benign_and_low_confidence_and_rejected(
    client: TestClient, image_bytes: ImageFactory
) -> None:
    benign = client.post("/api/v1/predict", files=upload(image_bytes((230, 40, 40)))).json()
    assert benign["risk_flag"] == "benign" and benign["predicted_class"] == "Acne"

    unsure = client.post("/api/v1/predict", files=upload(image_bytes((128, 128, 128)))).json()
    assert unsure["status"] == "low_confidence" and "uncertain" in unsure["summary"]

    dark = encode(np.full((64, 64, 3), 2, dtype=np.uint8))
    rejected = client.post("/api/v1/predict", files=upload(dark)).json()
    assert rejected["status"] == "rejected" and rejected["risk_flag"] == "uncertain"
    assert "too dark" in rejected["summary"]


def test_upload_validation(model_dir: Path, image_bytes: ImageFactory) -> None:
    with make_client(model_dir=model_dir, max_upload_mb=0.001) as small:
        assert_error(
            small.post("/api/v1/predict", files=upload(image_bytes(size=(200, 200)))),
            413,
            "image_too_large",
        )
    with make_client(model_dir=model_dir) as client:
        gif = encode(textured((1, 2, 3)), "GIF")
        assert_error(
            client.post("/api/v1/predict", files=upload(gif, "a.png")),
            415,
            "unsupported_image_type",
        )
        assert_error(
            client.post("/api/v1/predict", files=upload(b"not an image")), 400, "invalid_image"
        )
        assert_error(client.post("/api/v1/predict"), 422, "validation_error")


def test_oversized_body_is_refused_before_reading(model_dir: Path) -> None:
    with make_client(model_dir=model_dir, max_upload_mb=1) as client:
        body = b"x" * (1024 * 1024 + 128 * 1024)
        response = client.post(
            "/api/v1/predict",
            content=body,
            headers={"Content-Type": "multipart/form-data; boundary=b"},
        )
        assert_error(response, 413, "image_too_large")


def test_uploads_up_to_the_limit_stay_in_memory(model_dir: Path) -> None:
    from starlette.formparsers import MultiPartParser

    with make_client(model_dir=model_dir, max_upload_mb=8):
        assert MultiPartParser.spool_max_size >= 8 * 1024 * 1024


def test_explain_returns_pngs(client: TestClient, image_bytes: ImageFactory) -> None:
    response = client.post(
        "/api/v1/explain",
        params={"class_name": "Nevus", "alpha": 0.6},
        files=upload(image_bytes(size=(96, 64))),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["explained_class"] == "Nevus"
    for key in ("heatmap_png", "overlay_png"):
        image = Image.open(io.BytesIO(base64.b64decode(body[key])))
        assert image.format == "PNG" and image.size == (96, 64)


def test_explain_rejects_bad_parameters(client: TestClient, image_bytes: ImageFactory) -> None:
    data = image_bytes()
    assert_error(
        client.post("/api/v1/explain", params={"class_name": "Psoriasis"}, files=upload(data)),
        400,
        "unknown_class",
    )
    assert_error(
        client.post("/api/v1/explain", params={"alpha": 2}, files=upload(data)),
        422,
        "validation_error",
    )


def test_missing_model_keeps_service_up(tmp_path: Path, image_bytes: ImageFactory) -> None:
    with make_client(model_dir=tmp_path / "nothing") as client:
        assert client.get("/api/v1/health").status_code == 200
        ready = client.get("/api/v1/ready")
        assert ready.status_code == 503 and ready.json()["ready"] is False
        assert "Cannot read" in ready.json()["detail"]
        assert_error(
            client.post("/api/v1/predict", files=upload(image_bytes())), 503, "model_unavailable"
        )
        assert_error(client.get("/api/v1/meta"), 503, "model_unavailable")


def test_rate_limit(model_dir: Path) -> None:
    with make_client(model_dir=model_dir, rate_limit_per_minute=2) as client:
        assert client.get("/api/v1/health").status_code == 200
        assert client.get("/api/v1/health").status_code == 200
        limited = client.get("/api/v1/health")
        assert_error(limited, 429, "rate_limited")
        assert int(limited.headers["retry-after"]) >= 1


def test_rate_limit_uses_forwarded_ip_only_when_trusted(model_dir: Path) -> None:
    for trusted, expected in ((True, 200), (False, 429)):
        with make_client(
            model_dir=model_dir, rate_limit_per_minute=1, trust_forwarded_for=trusted
        ) as client:
            client.get("/api/v1/health", headers={"X-Forwarded-For": "10.0.0.1"})
            second = client.get("/api/v1/health", headers={"X-Forwarded-For": "10.0.0.2, 1.1.1.1"})
            assert second.status_code == expected


def test_cors_only_for_configured_origins(model_dir: Path) -> None:
    with make_client(model_dir=model_dir, allowed_origins=["https://demo.example"]) as client:
        allowed = client.get("/api/v1/health", headers={"Origin": "https://demo.example"})
        assert allowed.headers["access-control-allow-origin"] == "https://demo.example"
        other = client.get("/api/v1/health", headers={"Origin": "https://evil.example"})
        assert "access-control-allow-origin" not in other.headers


def test_request_id_is_echoed_and_unknown_routes_use_error_shape(client: TestClient) -> None:
    assert (
        client.get("/api/v1/health", headers={"X-Request-ID": "abc123"}).headers["x-request-id"]
        == "abc123"
    )
    assert_error(client.get("/api/v1/nope"), 404, "not_found")
    assert_error(client.get("/api/v1/predict"), 405, "method_not_allowed")


def test_unexpected_errors_are_hidden(client: TestClient, image_bytes: ImageFactory) -> None:
    class Broken:
        meta = client.app.state.predictor.meta  # type: ignore[attr-defined]

        def predict(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("secret internal detail")

    client.app.state.predictor = Broken()  # type: ignore[attr-defined]
    no_raise = TestClient(client.app, raise_server_exceptions=False)
    response = no_raise.post("/api/v1/predict", files=upload(image_bytes()))
    assert_error(response, 500, "internal_error")
    assert "secret" not in response.text


def test_serves_frontend_when_built(model_dir: Path, tmp_path: Path) -> None:
    static = tmp_path / "dist"
    static.mkdir()
    (static / "index.html").write_text("<h1>MedVision</h1>", encoding="utf-8")
    with make_client(model_dir=model_dir, static_dir=static) as client:
        assert "MedVision" in client.get("/").text
        assert client.get("/api/v1/health").status_code == 200


def test_openapi_documents_endpoints(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert {
        "/api/v1/predict",
        "/api/v1/explain",
        "/api/v1/health",
        "/api/v1/ready",
        "/api/v1/meta",
    } <= set(paths)
    assert client.get("/docs").status_code == 200


def test_access_log_is_json_without_image_content(
    client: TestClient, image_bytes: ImageFactory
) -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    access_logger = logging.getLogger("app.access")
    access_logger.addHandler(handler)
    try:
        client.post(
            "/api/v1/predict", files=upload(image_bytes()), headers={"X-Request-ID": "log-test"}
        )
    finally:
        access_logger.removeHandler(handler)
    lines = [json.loads(line) for line in stream.getvalue().splitlines()]
    access = [line for line in lines if line.get("request_id") == "log-test"]
    assert access and access[-1]["path"] == "/api/v1/predict" and access[-1]["status"] == 200
    assert isinstance(access[-1]["latency_ms"], float)
    assert all("PNG" not in json.dumps(line) for line in lines)
