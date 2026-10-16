# medvision-ml

Offline pipeline for the skin-lesion classifier: dataset checks and splitting,
two-stage transfer-learning training, evaluation reports and Grad-CAM.
TensorFlow is used only here; the serving backend never imports it.

## Setup

```bash
uv sync                      # from this folder; installs Python 3.11 if needed
uv run medvision-ml --help
```

Settings live in [`configs/skin.yaml`](configs/skin.yaml). Override any value
with an environment variable `MEDVISION__<SECTION>__<KEY>`, for example
`MEDVISION__TRAINING__EPOCHS=5`.

## Dataset

The model is trained on ISIC 2019; see [`docs/DATASETS.md`](../docs/DATASETS.md)
for download, licence and preparation steps.

## Dataset layout

```
data/skin/
├── Train/<class>/*.jpg
├── Validation/<class>/*.jpg
└── Test/<class>/*.jpg
```

Class indices follow the alphabetical order of the class folders. The dataset
is not part of the repository.

## Commands

| Command | What it does |
|---|---|
| `medvision-ml check-leakage` | Finds exact and near-duplicate images (difference hash) and lesions (`groups.csv`) that appear in more than one split, plus label conflicts and unreadable files. Exits with code 1 if leakage is found. |
| `medvision-ml prepare-isic2019 ...` | Lays out the ISIC 2019 release as class folders and writes `groups.csv` (image → lesion). |
| `medvision-ml split --source raw/` | Builds stratified 80/10/10 splits from a folder of class folders. Lesion groups (`groups.csv`) and near-duplicates are merged first so every photo of a lesion lands in the same split. |
| `medvision-ml train [--run-name NAME]` | Stage 1 trains the head on a frozen ImageNet backbone; stage 2 fine-tunes the top backbone layers (batch-norm stays frozen). Uses square-root class weights by default. |
| `medvision-ml evaluate --model PATH` | Writes `metrics.json`, a classification report, per-image predictions, a confusion matrix and a reliability diagram. Metrics include the malignant-group recall and calibration error used for model selection. |
| `medvision-ml gradcam --model PATH --image IMG` | Saves a Grad-CAM overlay for one image. |

The prototype model stores its preprocessing in a `Lambda` layer, which Keras
only loads in unsafe mode. Pass `--trust-legacy-model` to `evaluate` or
`gradcam` for that file only. Models trained here do not need it.

## Model layout

`image → augmentation → backbone → pool → dropout → logits → probabilities`

Because the head is global-average-pool followed by a single Dense layer,
Grad-CAM for class *c* equals `ReLU(Σₖ W[k, c] · Aₖ)` over the backbone feature
maps. `cam_from_features` computes that in NumPy, and the test suite checks it
against the TensorFlow Grad-CAM reference.

## Tests

```bash
uv run pytest            # includes a ~40 s end-to-end smoke test (marked slow)
uv run pytest -m "not slow and not data and not model"
```
