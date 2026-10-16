# MedVision AI

Explainable skin-lesion classification: upload a skin photo and get the most
likely conditions with calibrated confidence, a heatmap showing what the model
focused on, and a clear safety status.

> **Status:** planning and bootstrap. See [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md)
> for the architecture, roadmap, and design decisions.

## Planned features

- 8-class classifier trained on the ISIC 2019 dermoscopy dataset (Actinic
  Keratosis, Basal Cell Carcinoma, Benign Keratosis, Dermatofibroma, Melanocytic
  Nevus, Melanoma, Squamous Cell Carcinoma, Vascular Lesion) built on
  EfficientNetB0 transfer learning, with lesion-grouped splits to prevent leakage
- Grad-CAM visual explanations
- Calibrated probabilities, out-of-distribution rejection, and a
  malignancy-risk flag
- FastAPI backend served with ONNX Runtime and a React frontend (drag and
  drop or camera upload, risk badges, heatmap with adjustable opacity)
- One-command Docker run, CI, and a public demo (planned)

## Train the model

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/OmkarJayvantJadhav/MedVisionAI/blob/main/notebooks/train_colab.ipynb)

The notebook trains on ISIC 2019 with a free Colab GPU, checks the splits for
leakage, evaluates on the held-out test set and exports the serving files to
Google Drive. See [`docs/DATASETS.md`](docs/DATASETS.md) for the data and its
licence.

## Development

Requirements: [uv](https://docs.astral.sh/uv/) and GNU Make. Python 3.11 is
installed by uv if missing.

```bash
make setup   # install dependencies for ml/ and backend/ and the git hooks
make check   # lint, type check and test the Python packages (same as CI)
make frontend-setup frontend-check   # install, lint, test and build the frontend
make run            # API on http://localhost:8000 (needs model files in models/)
make frontend-dev   # UI on http://localhost:5173, proxied to the API
make help    # list all targets
```

| Folder | Contents |
|---|---|
| `ml/` | training, evaluation, calibration and ONNX export (TensorFlow) |
| `backend/` | FastAPI inference service (ONNX Runtime, no TensorFlow) |
| `frontend/` | React (JavaScript) + Vite + Tailwind web app |
| `models/` | downloaded model artifacts (not committed) |
| `docs/` | project plan, dataset sources and licences, design documents |
| `notebooks/` | Colab notebook that trains, evaluates and exports the model |

## Disclaimer

This project is for education and screening research only. It is **not** a
medical device and does not provide a diagnosis. Always consult a qualified
dermatologist.
