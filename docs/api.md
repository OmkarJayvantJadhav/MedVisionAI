# API reference (v1)

Interactive documentation is served at `/docs` (Swagger UI) and the schema at
`/openapi.json`. All endpoints live under `/api/v1`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness: `{"status": "ok"}` |
| GET | `/ready` | Readiness: `200` with the model version once the model is loaded and verified, otherwise `503` with the reason |
| GET | `/meta` | Model version, classes, malignant classes, thresholds, upload limit, disclaimer |
| POST | `/predict` | Multipart field `file` (JPEG, PNG or WebP): prediction |
| POST | `/explain` | Same as `/predict`, plus a Grad-CAM heatmap and overlay. Query: `class_name` (default: top class), `alpha` (0–1, overlay opacity) |

## Prediction response

```json
{
  "status": "low_confidence",
  "predicted_class": "Melanoma",
  "confidence": 0.23,
  "top_k": [{"class_name": "Melanoma", "probability": 0.23}, "..."],
  "risk_flag": "malignant_suspected",
  "malignant_probability": 0.72,
  "reasons": ["low_confidence", "malignant_probability_above_threshold"],
  "model_version": "skin-effb0-v1.0.0",
  "calibrated": false,
  "thresholds_fitted": false,
  "summary": "The model is uncertain. Its best guess is Melanoma (23%) ...",
  "disclaimer": "Not a medical diagnosis. ...",
  "latency_ms": 46.1
}
```

- `status`: `ok`, `low_confidence` (top probability below the threshold) or
  `rejected` (too dark, too bright, too blurry, or out of distribution).
- `risk_flag`: `malignant_suspected` when the summed probability of the
  malignant and pre-malignant classes reaches its threshold, even if a benign
  class ranks first; `uncertain` for rejected or low-confidence images;
  otherwise `benign`.
- `/explain` adds `explained_class`, `heatmap_png` and `overlay_png`
  (base64-encoded PNGs at display size).

## Errors

Every error uses the same body: `{"error": {"code": "...", "message": "..."}}`.

| Status | Code | When |
|---|---|---|
| 400 | `invalid_image` | The file cannot be decoded, or is smaller than 32 px |
| 400 | `unknown_class` | `class_name` is not a model class |
| 413 | `image_too_large` | Over the upload limit (default 8 MB) or 4096 px per side |
| 415 | `unsupported_image_type` | Not JPEG, PNG or WebP (detected from content) |
| 422 | `validation_error` | Missing file or invalid query parameter |
| 429 | `rate_limited` | Over the per-client limit (default 30 requests/min on `/api/`); see `Retry-After` |
| 503 | `model_unavailable` | The model is not loaded |
| 500 | `internal_error` | Unexpected failure; details are logged, never returned |

Every response carries an `X-Request-ID` header (echoed if the client sent one).

## Configuration

Environment variables (or a `.env` file in `backend/`):

| Variable | Default | Meaning |
|---|---|---|
| `MEDVISION_MODEL_DIR` | `<repo>/models` | Folder with `model.onnx`, `head_weights.npz`, `model_meta.json` |
| `MEDVISION_VERIFY_CHECKSUMS` | `true` | Verify SHA-256 of model files at startup |
| `MEDVISION_ALLOWED_ORIGINS` | `[]` | CORS origins, as a JSON list |
| `MEDVISION_MAX_UPLOAD_MB` | `8` | Upload size limit |
| `MEDVISION_RATE_LIMIT_PER_MINUTE` | `30` | Per-client limit on `/api/`; `0` disables it |
| `MEDVISION_TRUST_FORWARDED_FOR` | `false` | Take the client IP from `X-Forwarded-For` (only behind a trusted proxy) |
| `MEDVISION_STATIC_DIR` | unset | Built frontend to serve at `/` |
| `MEDVISION_LOG_LEVEL` | `INFO` | Log level; logs are JSON lines on stdout |

Uploaded images are processed in memory only and never logged. Requests whose
`Content-Length` exceeds the upload limit are refused before the body is read,
and multipart uploads up to the limit are kept in memory rather than spooled to
a temporary file.
