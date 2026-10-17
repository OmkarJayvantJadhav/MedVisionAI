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

## Model

Release [`model-v2.0.0`](https://github.com/OmkarJayvantJadhav/MedVisionAI/releases/tag/model-v2.0.0)
(EfficientNetB0, ISIC 2019, CC BY-NC 4.0). Results on the held-out test split
(2,527 images, lesion-grouped, no leakage), reproducible with
`medvision-ml evaluate` and `medvision-ml calibrate`:

| Metric | Value |
|---|---|
| Accuracy (top-1) | 0.708 |
| Macro F1 | 0.508 |
| Melanoma recall (top-1) | 0.435 |
| Malignant-group recall at the fitted threshold | 0.904 (precision 0.681) |
| Calibration error (ECE) | 0.026 |

Top-1 melanoma recall is still low, so the app flags a possible malignancy
from the summed probability of all malignant classes, with a threshold fitted
to catch 90% of malignant validation images.

```bash
make download-model BASE_URL=https://github.com/OmkarJayvantJadhav/MedVisionAI/releases/download/model-v2.0.0
```

## Train the model

Training runs through the `medvision-ml` command line: prepare and split
ISIC 2019, check the splits for leakage, train, evaluate on the held-out test
set and export the serving files. See [`docs/DATASETS.md`](docs/DATASETS.md)
for the steps, the data and its licence.

[`notebooks/train_kaggle.ipynb`](notebooks/train_kaggle.ipynb) runs every step
on a free Kaggle GPU in the background (*Save & Run All*), so no browser tab
has to stay open.

## Run with Docker

```bash
docker compose up --build    # then open http://localhost:7860
```

The image bundles the website, the API and the checksum-verified model. See
[`docs/deploy.md`](docs/deploy.md) for the free Render deployment.

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
| `notebooks/` | Kaggle notebook that runs the full training pipeline on a free GPU |
| `docs/` | project plan, dataset sources and licences, design documents |

## Disclaimer

This project is for education and screening research only. It is **not** a
medical device and does not provide a diagnosis. Always consult a qualified
dermatologist.
