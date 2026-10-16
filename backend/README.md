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
| `artifacts.py` | Reads `model_meta.json`, verifies SHA-256 checksums, loads the head weights |
| `calibration.py` | Recomputes logits from the feature maps and head weights, applies temperature scaling, computes the energy score |
| `quality.py` | Brightness and sharpness (Laplacian variance) checks |
| `decision.py` | Turns probabilities into a `status` (`ok`, `low_confidence`, `rejected`) and a `risk_flag` (`malignant_suspected`, `benign`, `uncertain`) |
| `cam.py` | Grad-CAM as `ReLU(feature_maps @ W[:, c])`, colouring and overlay |
| `predictor.py` | `Predictor.predict(bytes, explain=...)`: the single entry point |
| `download.py` | Fetches model artifacts from a release URL and verifies checksums |

Thresholds and the calibration temperature ship in `model_meta.json`. Until
they are fitted on validation data, responses report `calibrated: false` and
`thresholds_fitted: false`.

## Try it

```bash
uv sync
# Put model.onnx, head_weights.npz and model_meta.json in ../models (see `make download-model`)
uv run python -m app.inference.cli path/to/image.jpg --overlay-dir overlays/
```

On a laptop CPU a prediction takes about 16 ms and a prediction with an
explanation about 75 ms (EfficientNetB0, 224 px).

## Tests

```bash
uv run pytest
```

The tests build a tiny ONNX model in memory with the same structure as the real
one, so they need no model download. Parity with TensorFlow (preprocessing,
probabilities, Grad-CAM) is tested in `ml/tests/test_export_parity.py`.
