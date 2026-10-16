# MedVision AI

Explainable skin-lesion classification: upload a skin photo and get the most
likely conditions with calibrated confidence, a heatmap showing what the model
focused on, and a clear safety status.

> **Status:** planning and bootstrap. See [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md)
> for the architecture, roadmap, and design decisions.

## Planned features

- 8-class classifier (Acne, Actinic Keratosis, Basal Cell Carcinoma,
  Dermatofibroma, Melanocytic Nevus, Melanoma, Seborrheic Keratosis,
  Squamous Cell Carcinoma) built on EfficientNetB0 transfer learning
- Grad-CAM visual explanations
- Calibrated probabilities, out-of-distribution rejection, and a
  malignancy-risk flag
- FastAPI backend served with ONNX Runtime, React frontend, one-command
  Docker run, CI, and a public demo

## Development

Requirements: [uv](https://docs.astral.sh/uv/) and GNU Make. Python 3.11 is
installed by uv if missing.

```bash
make setup   # install dependencies for ml/ and backend/ and the git hooks
make check   # lint, type check and test everything (same as CI)
make help    # list all targets
```

| Folder | Contents |
|---|---|
| `ml/` | training, evaluation, calibration and ONNX export (TensorFlow) |
| `backend/` | FastAPI inference service (ONNX Runtime, no TensorFlow) |
| `models/` | downloaded model artifacts (not committed) |
| `docs/` | project plan and design documents |

## Disclaimer

This project is for education and screening research only. It is **not** a
medical device and does not provide a diagnosis. Always consult a qualified
dermatologist.
