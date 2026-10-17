# medvision-backend

Serves predictions and Grad-CAM explanations for the skin-lesion classifier.
Inference uses ONNX Runtime and NumPy only; TensorFlow is never installed here.

## Run the API

```bash
uv sync
MEDVISION_MODEL_DIR=../models uv run uvicorn app.main:create_app --factory --port 8000
# or, from the repository root: make run
curl -F file=@lesion.jpg http://localhost:8000/api/v1/predict
```

Open http://localhost:8000/docs for interactive documentation. Endpoints,
response fields, error codes and configuration are described in
[`docs/api.md`](../docs/api.md).

| Package | Responsibility |
|---|---|
| `app/main.py` | App factory; loads the model once at startup and keeps the service up (with `/ready` = 503) if it cannot |
| `app/api/v1/` | Routes; CPU-bound inference runs in a thread pool |
| `app/core/` | Settings, JSON logging, error mapping, request IDs, rate limiting |
| `app/schemas/` | Pydantic response models |
| `app/services/summary.py` | Plain-language summaries that only restate the model output |

## Inference core (`app/inference/`)

| Module | Responsibility |
|---|---|
| `preprocessing.py` | Decodes JPEG/PNG/WebP by content (not file name), enforces size limits, applies EXIF orientation, and resizes exactly like the training pipeline (`tf.image.resize`, bilinear, half-pixel centres) |
| `artifacts.py` | Reads `model_meta.json`, verifies the SHA-256 of every listed file, loads the head weights and, when the model has one, the out-of-distribution statistics (`ood_stats.npz`) |
| `calibration.py` | Pools the feature maps, recomputes the logits, applies temperature scaling and computes the Mahalanobis distance used by the out-of-distribution check |
| `quality.py` | Brightness, sharpness (Laplacian variance) and sensor-noise checks |
| `decision.py` | Applies the safety rules: rejection reasons, then a `status` (`ok`, `low_confidence`, `rejected`) and a `risk_flag` (`malignant_suspected`, `benign`, `uncertain`) |
| `cam.py` | Grad-CAM as `ReLU(feature_maps @ W[:, c])`, colouring and overlay |
| `predictor.py` | `Predictor.predict(bytes, explain=...)`: the single entry point |
| `download.py` | Fetches model artifacts from a release URL and verifies checksums |

The calibration temperature and every threshold ship in `model_meta.json`.
Models without fitted values report `calibrated: false` or
`thresholds_fitted: false`, and the out-of-distribution and noise checks stay
off when their values are missing.

## Try it

```bash
uv sync
# Put the release files in ../models first (see `make download-model`)
uv run python -m app.inference.cli path/to/image.jpg --overlay-dir overlays/
```

On a typical laptop CPU a prediction takes a few tens of milliseconds, a little
more with an explanation (EfficientNet-B0, 224 px); `make smoke` reports p95.

## Tests

```bash
uv run pytest
```

The tests build a tiny ONNX model in memory with the same structure as the real
one, so they need no model download. Parity with TensorFlow (preprocessing,
probabilities, Grad-CAM, out-of-distribution distance and noise estimate) is
tested in `ml/tests/test_export_parity.py`.
