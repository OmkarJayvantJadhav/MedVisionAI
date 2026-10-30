<div align="center">

# MedVision AI

**A skin-cancer referral check for dermoscopy images.**

MedVision AI helps a clinician with a dermatoscope answer one question about a skin
lesion: *should it be referred to a dermatologist?* It flags 9 in 10 malignant lesions at
its operating point, gives calibrated probabilities, shows where it looked, and refuses
images it cannot judge. The clinician makes the decision.

[![Open the app](https://img.shields.io/badge/Open_the_app-medvision--ai-0b6e75?style=for-the-badge)](https://medvision-ai-hguo.onrender.com)
[![Model card](https://img.shields.io/badge/Model_card-read-12784a?style=for-the-badge)](docs/model_card.md)
[![Architecture](https://img.shields.io/badge/Architecture-docs-4a5a68?style=for-the-badge)](docs/architecture.md)

[![CI](https://github.com/OmkarJayvantJadhav/MedVisionAI/actions/workflows/ci.yml/badge.svg)](https://github.com/OmkarJayvantJadhav/MedVisionAI/actions/workflows/ci.yml)
[![Live site](https://github.com/OmkarJayvantJadhav/MedVisionAI/actions/workflows/live.yml/badge.svg)](https://github.com/OmkarJayvantJadhav/MedVisionAI/actions/workflows/live.yml)
![Python](https://img.shields.io/badge/Python-3.11-3776ab?logo=python&logoColor=white)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-CPU-005ced?logo=onnx&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-19-149eca?logo=react&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-training_only-ff6f00?logo=tensorflow&logoColor=white)
[![Code licence: MIT](https://img.shields.io/badge/Code-MIT-0b6e75)](LICENSE)
![Model licence](https://img.shields.io/badge/Model-CC_BY--NC_4.0-lightgrey)

<br>

<!-- <img src="docs/assets/demo.gif" alt="Demo: a melanoma sample is chosen, the photo is scanned, the result shows a malignancy warning with top matches, and the heatmap opacity is adjusted" width="860">

<sub>Choose a photo, watch it being analysed, read the verdict, then explore the heatmap.
The sample is an ISIC 2019 test image the model never saw in training.</sub> -->

</div>

> [!IMPORTANT]
> **Decision support, not a diagnosis and not a medical device.** MedVision AI has not been
> clinically validated. It supports a clinician's referral decision and never replaces it.
> Anyone worried about a skin lesion should see a doctor.

> [!NOTE]
> The hosted app sleeps after 15 idle minutes, so the first request can take about a minute
> while it wakes up. After that, an analysis takes one to three seconds on the hosted CPU
> share (about 30 ms on a normal machine).

---

## Contents

- [Who it is for](#who-it-is-for)
- [The problem](#the-problem)
- [Objective and success criteria](#objective-and-success-criteria)
- [Scope](#scope)
- [What it does](#what-it-does)
- [Using the app](#using-the-app)
- [How it works](#how-it-works)
- [Architecture at a glance](#architecture-at-a-glance)
- [Safety by design](#safety-by-design)
- [Results](#results)
- [Engineering highlights](#engineering-highlights)
- [Tech stack](#tech-stack)
- [Quick start](#quick-start)
- [API](#api)
- [Training and evaluation](#training-and-evaluation)
- [Repository layout](#repository-layout)
- [Quality gates](#quality-gates)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Documentation](#documentation)
- [Acknowledgements and licence](#acknowledgements-and-licence)

---

## Who it is for

| User | Situation |
|---|---|
| **General practitioners and primary-care nurses** with a dermatoscope | A patient shows a mole or spot; the clinician decides whether to refer |
| **Teledermatology triage** | Dermoscopy images arrive from other sites and need sorting by urgency |
| **Clinicians in training** | Learning how dermoscopic features relate to malignancy, with a heatmap to compare against |
| **Researchers and engineers** | A reproducible, documented baseline for safe skin-lesion classification |

**Not intended for the public.** Self-checks need a dermatoscope, which most people do not
have; ordinary phone photos are refused by design.

## The problem

Skin cancer is curable when it is caught early, and most suspicious lesions are first seen
in primary care. There, clinicians decide which lesions to refer to a dermatologist. Refer
too few and a melanoma is delayed; refer too many and dermatology waiting lists grow for
lesions that were benign all along. Dermoscopy improves that decision, but it takes
training and experience to read.

Image classifiers trained on dermoscopy could support the decision, but most published ones
are not safe to use for it:

- **They answer everything.** A clinical photo, a blurry image or a picture of something
  else still gets a confident answer.
- **They hide their misses.** A headline accuracy says nothing about how many cancers go
  unflagged, or for whom.
- **Their confidence means nothing.** "92%" is a raw score, not a probability.
- **They answer the wrong question.** A single top-1 diagnosis can say "nevus" while the
  combined probability of malignancy is high enough to refer.
- **They are black boxes.** The clinician cannot see what the model looked at.

## Objective and success criteria

Build an **open, reproducible referral check** for dermoscopy images of a single lesion.
For each image it returns one of four answers, always in this order of priority:
**refer** (possible malignancy), **unsure**, **no flag**, or **not assessed**, with
calibrated probabilities and a heatmap. It supports the clinician's referral decision; it
never makes a diagnosis.

| # | Objective | Success criterion | Status (model-v2.2.0, test split) |
|---|---|---|---|
| 1 | **Catch cancers** | Malignant-group recall ≥ 0.90 at the referral threshold, with the referral rate it causes reported | ✅ 0.904, 49% of test images flagged |
| 2 | **Honest confidence** | Calibration error (ECE) ≤ 0.05 | ✅ 0.026 |
| 3 | **Refuse what it cannot judge** | ≥ 99% of non-dermoscopy images rejected while ≥ 98% of lesion images pass | ✅ 99.6% rejected, 98.4% pass |
| 4 | **No silent failures** | Damaged images are rejected rather than silently misread | ✅ noise: 0% silent misses · ⚠️ JPEG quality 10: 25% |
| 5 | **Explain itself** | A heatmap for every answer, matching TensorFlow Grad-CAM (cosine ≥ 0.99) | ✅ parity test in CI |
| 6 | **Report its weaknesses** | Robustness and subgroup results regenerated by one command | ✅ `make evaluate` and the [model card](docs/model_card.md) |
| 7 | **Run anywhere** | CPU-only inference in one container under 512 MB, no external calls or API keys | ✅ about 30 ms per image, about 170 MB of memory |
| 8 | **Classify well** | Macro F1 ≥ 0.72 and melanoma top-1 recall ≥ 0.70, which also lowers the referral rate | ❌ 0.508 and 0.435: next on the [roadmap](#roadmap) |

The objectives are ranked. A change that improves accuracy but lowers cancer recall, or
makes the model more confident or less honest, is not accepted.

## Scope

**In scope**

- **Dermoscopy images of a single skin lesion**, taken with a dermatoscope: JPEG, PNG or
  WebP, up to 8 MB.
- **One decision:** whether the lesion should be referred to a dermatologist, expressed as
  refer, unsure, no flag or not assessed.
- **Supporting detail:** calibrated probabilities for the 8 ISIC 2019 lesion types (actinic
  keratosis, basal cell carcinoma, benign keratosis, dermatofibroma, melanocytic nevus,
  melanoma, squamous cell carcinoma, vascular lesion), the combined malignancy probability
  against the referral threshold, a heatmap and a fixed summary.
- **Anonymous, stateless use:** the image is processed in memory and discarded.

**Out of scope, by design**

| Not included | Why |
|---|---|
| Diagnosis, treatment or management advice | The output supports a referral decision; the diagnosis belongs to the dermatologist |
| Self-checks by the public, ordinary phone or clinical photos | The model only learned from dermoscopy; other images are refused rather than guessed at |
| Rashes, infections, inflammatory and other non-lesion conditions | Different problem, different data; they are not lesions to triage for cancer |
| Rare skin cancers outside the 8 classes (for example Merkel cell carcinoma) | No training data; the model cannot recognise them, so a "no flag" never rules them out |
| Text generated by a language model | The summary restates the model's output from a fixed template, so it can never invent a claim |
| Accounts, patient records or stored images | No patient data is kept anywhere |
| Commercial use | The training data, and so the model weights, are licensed for non-commercial use only |

## What it does

| | |
|---|---|
| **Flags for referral** | A referral flag when the four malignant and pre-malignant classes together pass a fitted threshold, even if a benign class ranks first |
| **Calibrates** | Temperature scaling makes a "70%" mean right about 70% of the time (calibration error 0.026) |
| **Shows its working** | The closest lesion types with calibrated probabilities, and a gauge of the combined malignancy probability with the referral threshold marked |
| **Rejects** | Non-dermoscopy images (out-of-distribution check), grainy images (noise check), and dark, washed-out or blurry images (quality checks) |
| **Explains** | A heatmap overlay with an opacity slider, computed from the same forward pass |
| **Respects privacy** | Images are processed in memory and never stored or logged; no accounts, no tracking, self-hosted fonts |
| **Runs on modest hardware** | EfficientNet-B0 in ONNX (16 MB) on a CPU, about 30 ms per image; one container, no external calls |

## Using the app

1. **Add a dermoscopy image.** Drop it on the card, browse your files or paste it with
   Ctrl+V. No image to hand? Pick one of the three samples (held-out ISIC 2019 test images).
2. **Watch it being analysed.** The image appears straight away, with a scanning line, the
   three server-side steps (quality validation, model inference, heatmap) and the elapsed
   time. The header shows whether the model is ready.
3. **Read the answer first.** One of four outcomes leads the result, always in this order of
   priority:

   | Answer | When | What it means for the clinician |
   |---|---|---|
   | **Refer: possible malignancy** | Malignant and pre-malignant classes together pass the referral threshold | Consider referral to a dermatologist |
   | **Unsure: low confidence** | Top confidence below the fitted threshold | Retake the image if it is poor; refer on clinical grounds if the lesion is new, changing or bleeding |
   | **No malignancy flag** | Neither of the above | The threshold was not reached; this does not rule out malignancy |
   | **Not assessed: image rejected** | A quality or out-of-distribution check failed | What to fix ("too grainy"), or that only dermoscopy images can be assessed |

4. **Look closer.** Below the answer: the clinical summary, the combined malignancy gauge
   with the referral threshold marked, the closest lesion type with the top three calibrated
   probabilities, the model version and the processing time. On the image, the heatmap shows
   where the model's attention was; toggle it or change its opacity. It is not a diagnosis
   or an outline of cancer.
5. **Technical details**, folded away, show the raw status, risk flag and reasons.

The interface is one workspace card that changes with the state rather than a dashboard of
panels, with background information in collapsible sections below it. It works in light and
dark mode (following the system until you choose), by keyboard and with a screen reader, and
on a phone or tablet, where the answer comes before the image.

## How it works

```mermaid
flowchart LR
    U(["Dermoscopy image"]) --> D["Decode and validate<br/>type, size, orientation"]
    D --> R["Resize to 224 px<br/>same as training"]
    R --> Q{"Quality checks<br/>brightness, sharpness, noise"}
    Q -- fail --> X(["Rejected:<br/>what to fix"])
    Q -- pass --> M["EfficientNet-B0<br/>ONNX Runtime on CPU"]
    M --> F[("Feature maps<br/>7 x 7 x 1280")]
    F --> O{"Out-of-distribution<br/>Mahalanobis distance"}
    O -- far --> Y(["Rejected:<br/>not a dermoscopy image"])
    O -- near --> L["Logits and<br/>temperature scaling"]
    L --> S{"Safety rules"}
    S --> V(["Referral answer, top 3,<br/>malignancy gauge"])
    F --> C["Class activation map<br/>NumPy, no TensorFlow"]
    C --> H(["Heatmap overlay"])
```

1. **Validate.** The upload is decoded in memory, never written to disk. Only real
   JPEG, PNG or WebP images up to 8 MB and 4096 px pass.
2. **Preprocess.** A NumPy re-implementation of TensorFlow's bilinear resize, so the
   served model sees exactly what it saw in training (covered by a parity test).
3. **Gate.** Brightness, sharpness and a sensor-noise estimate reject unusable photos.
4. **Classify.** One ONNX Runtime call returns the feature maps. The logits, the
   out-of-distribution distance and the heatmap are all derived from that one result.
5. **Decide.** Calibrated probabilities pass through the safety rules below and become one
   referral answer. The response always carries the disclaimer and a fixed summary.

## Architecture at a glance

```mermaid
flowchart LR
    subgraph Offline["Offline, GPU (Kaggle)"]
        direction TB
        T["ml/ pipeline (TensorFlow)<br/>split, train, calibrate,<br/>fit gates, evaluate, export"]
    end
    T --> R[("GitHub release<br/>ONNX model, head weights,<br/>gate statistics, metadata")]
    subgraph Online["Online, CPU only (Render)"]
        direction TB
        W["React app"] --> A["FastAPI"]
        A --> I["ONNX Runtime + NumPy<br/>gates, calibration, heatmap"]
    end
    R -- "downloaded and checksum-verified<br/>when the image is built" --> I
```

The two halves share nothing at runtime except a versioned, checksum-verified model
release. Components, the life of a request and the key design decisions are in
[docs/architecture.md](docs/architecture.md).

## Safety by design

Every threshold is **fitted on the validation split** and **reported on the untouched test
split**. All of them ship inside the versioned model release (`model_meta.json`).

| Rule | How it is set | Effect on the 2,527 test images |
|---|---|---|
| **Referral flag**: melanoma + BCC + SCC + actinic keratosis probability ≥ threshold | Catch 90% of malignant validation images | **90.4%** of malignant cases flagged, precision 68.1%, 49% of images flagged |
| **Low confidence**: top probability below threshold | Confident answers must be ≥ 85% accurate on validation | 59% answered confidently, 86.9% of those correct |
| **Out-of-distribution**: Mahalanobis distance of the image features from the training images | 99% of validation images pass | 98.9% of lesions pass; **99.6%** of 497 everyday photos rejected |
| **Noise check**: sensor noise measured in the image's smoothest patches | 99% of validation images pass | Cancers missed silently on noisy photos: **49% → 0%** |
| **Quality checks**: brightness and sharpness | Fixed, conservative bounds | Black, washed-out and featureless images rejected |

The answer order is deliberate. A referral flag outranks "unsure", which outranks
"no flag", so a warning is never hidden behind uncertainty. A rejected image gets an
explanation of *why* ("too grainy", "not a dermoscopy image") instead of a guess.

## Results

Model release [`model-v2.2.0`](https://github.com/OmkarJayvantJadhav/MedVisionAI/releases/tag/model-v2.2.0):
EfficientNet-B0 trained on ISIC 2019, evaluated on a **lesion-grouped, leakage-checked** test
split of 2,527 images. Every number here is regenerated by `make evaluate`.

<table>
<tr><td valign="top">

**Overall**

| Metric | Value |
|---|---|
| Accuracy (top-1) | 0.708 |
| Macro F1 | 0.508 |
| ROC-AUC (one-vs-rest) | 0.935 |
| Malignant-group ROC-AUC | 0.916 |
| Calibration error (ECE) | 0.026 |
| Malignant recall at the referral flag | **0.904** |

</td><td valign="top">

**Per class (recall)**

| Class | Recall | Test images |
|---|---|---|
| Melanocytic nevus | 0.90 | 1,286 |
| Benign keratosis | 0.68 | 261 |
| Basal cell carcinoma | 0.62 | 331 |
| Vascular lesion | 0.52 | 25 |
| Melanoma | 0.43 | 451 |
| Actinic keratosis | 0.30 | 86 |
| Dermatofibroma | 0.29 | 24 |
| Squamous cell carcinoma | 0.17 | 63 |

</td></tr>
</table>

**Robustness.** The share of malignant images missed *without any warning* under image
damage, with all checks active:

| Clean | Mild blur | Phone-like | Darker | JPEG q30 | Noise σ8 | Noise σ20 | JPEG q10 |
|---|---|---|---|---|---|---|---|
| 9.3% | 7.7% | 9.9% | 13.6% | 12.1% | **0%** | **0%** | 25.1% |

**By patient group.** Referral-flag recall: men 0.95, women 0.84, ages 30–49 0.76,
ages 70+ 0.95, lower limbs 0.82. Skin type is not recorded in ISIC 2019, so performance
on darker skin is **unknown**. The [model card](docs/model_card.md) states this plainly.

<details>
<summary><b>Calibration and confusion matrix</b></summary>
<br>

| Before temperature scaling | After |
|---|---|
| <img src="docs/assets/reliability_test_before.png" width="380" alt="Reliability diagram before calibration"> | <img src="docs/assets/reliability_test_after.png" width="380" alt="Reliability diagram after calibration"> |

<p align="center"><img src="docs/assets/confusion_matrix_test.png" width="640" alt="Test-set confusion matrix"></p>

</details>

## Engineering highlights

<details open>
<summary><b>Grad-CAM without TensorFlow, from the same forward pass</b></summary>

For a `pooling → dense` head, Grad-CAM's channel weights are exactly the dense layer's
weights, so the heatmap reduces to a weighted sum of the last feature maps. The ONNX
export outputs those feature maps, and the backend computes the heatmap in NumPy. That
removes TensorFlow from serving and makes an explanation nearly free. A parity test
compares the result with TensorFlow Grad-CAM (cosine similarity ≥ 0.99).
</details>

<details open>
<summary><b>Rejection gates measured by the failure that matters</b></summary>

A rejected photo is a safe outcome; a missed cancer with no warning is not. The robustness
suite therefore counts **silent misses**. That metric showed noisy photos slipping past the
out-of-distribution check and losing up to half of malignant cases. A noise estimate taken
in each image's smoothest patches, so hair and texture don't count, closed the gap
completely at a cost of 0.5% of clean images.
</details>

<details open>
<summary><b>Leakage-free splits, and the bug that almost hid it</b></summary>

16,459 of the 25,331 ISIC images belong to lesions photographed several times, so splits
are grouped by lesion and by near-duplicate image hash. The first hash mostly encoded the
dark circular border common to dermoscopy photos. It chained 3,676 unrelated images into
one "duplicate" group and skewed the test set. Hashing the centre of the image fixed it,
and `split` now refuses any group larger than 20 images.
</details>

<details>
<summary><b>More decisions worth a look</b></summary>

- **Float32 rebuild bug.** A mixed-precision model reloaded through `set_weights` silently
  lost its input normalisation (validation 0.695 → 0.646). Loading through Keras's own
  weight loader fixed it, and a regression test guards it.
- **Out-of-distribution score.** Energy and max-softmax scores were tried first, but only
  rejected 18% of everyday photos at 99% lesion acceptance. Mahalanobis distance on the
  pooled features reached AUROC 0.9998.
- **Hosting.** The app first targeted Hugging Face Docker Spaces, then moved to Render from
  a Blueprint (`render.yaml`). A daily workflow smoke-tests the public URL.
- **One card, not a dashboard.** The first interface showed the photo twice (preview and
  heatmap) across a page of boxes. One card that turns from drop zone into photo-plus-result
  keeps the answer in focus, and phones get the verdict above the photo.
- **Typed contract without TypeScript.** The frontend is plain JavaScript by choice. The
  committed OpenAPI schema keeps it honest: frontend tests validate their mocks against it,
  and a backend test fails if it goes stale.

Every decision, with its reasoning, is in the [decision log](docs/PROJECT_PLAN.md#12-decision-log).
</details>

## Tech stack

| Layer | Choices |
|---|---|
| **Model** | EfficientNet-B0 (ImageNet weights), TensorFlow / Keras 3, two-stage fine-tuning, square-root class weights, mixed precision on a Kaggle T4 GPU |
| **Export** | tf2onnx (opset 17) with two outputs, `probabilities` and `feature_maps`; dense-head weights, gate statistics and SHA-256 checksums shipped with each release |
| **Serving** | FastAPI, Pydantic v2, ONNX Runtime (CPU) and NumPy, with **no TensorFlow**; structured JSON logs, per-client rate limit, restricted CORS |
| **Frontend** | React 19 (JavaScript), Vite, Tailwind CSS v4, self-hosted fonts, Vitest |
| **Ops** | Multi-stage Docker image (non-root, health check, model verified at build), GitHub Actions CI, Render, daily live smoke test |
| **Tooling** | uv with lockfiles, ruff, mypy (strict), ESLint, Prettier, pre-commit, Make |

## Quick start

**Run the whole app** (website, API and the checksum-verified model) with Docker:

```bash
git clone https://github.com/OmkarJayvantJadhav/MedVisionAI.git
cd MedVisionAI
docker compose up --build        # open http://localhost:7860
```

**Develop locally** with [uv](https://docs.astral.sh/uv/), Node.js and GNU Make:

```bash
make setup                       # Python dependencies for ml/ and backend/, git hooks
make download-model BASE_URL=https://github.com/OmkarJayvantJadhav/MedVisionAI/releases/download/model-v2.2.0
make run                         # API on http://localhost:8000 (docs at /docs)
make frontend-setup frontend-dev # UI on http://localhost:5173, proxied to the API
make check frontend-check        # everything CI runs
make help                        # every target
```

## API

Interactive documentation is served at `/docs`. All endpoints live under `/api/v1`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness |
| `GET` | `/ready` | Model loaded and checksum-verified |
| `GET` | `/meta` | Model version, classes, fitted thresholds, disclaimer |
| `POST` | `/predict` | Prediction for one image (`multipart/form-data`, field `file`) |
| `POST` | `/explain` | Prediction plus heatmap and overlay PNGs |

```bash
curl -s -F file=@frontend/public/samples/melanoma.jpg \
  https://medvision-ai-hguo.onrender.com/api/v1/predict
```

```jsonc
{
  "status": "ok",                       // ok | low_confidence | rejected
  "predicted_class": "Melanoma",
  "confidence": 0.91,
  "top_k": [{ "class_name": "Melanoma", "probability": 0.91 }, "..."],
  "risk_flag": "malignant_suspected",   // malignant_suspected | benign | uncertain
  "malignant_probability": 0.91,
  "reasons": ["malignant_probability_above_threshold"],
  "model_version": "skin-effb0-v2.2.0",
  "calibrated": true,
  "summary": "…plain-language text…",
  "disclaimer": "Not a medical diagnosis. …"
}
```

Full reference, error codes and configuration: [docs/api.md](docs/api.md).

## Training and evaluation

Everything runs through one tested command-line tool, `medvision-ml`. The notebook only
calls it.

```mermaid
flowchart LR
    A["prepare-isic2019"] --> B["split<br/>lesion and duplicate groups"]
    B --> C["check-leakage"]
    C --> D["train"]
    D --> E["evaluate"]
    E --> F["calibrate"]
    F --> G["fit-ood<br/>and noise threshold"]
    G --> H["export<br/>ONNX, weights, gate stats"]
    H --> I[("GitHub release<br/>model-vX.Y.Z")]
    G --> J["robustness"]
    G --> K["subgroups"]
```

- **Train on Kaggle:** [`notebooks/train_kaggle.ipynb`](notebooks/train_kaggle.ipynb) runs
  the full pipeline on a Kaggle GPU in the background (*Save & Run All*). No tab has to stay open.
- **Reproduce every reported number:**
  `make evaluate MODEL=ml/artifacts/<run>/model.keras METADATA=<ISIC metadata CSV>`.
- **Data, licences and steps:** [docs/DATASETS.md](docs/DATASETS.md).

## Repository layout

```text
MedVisionAI/
├── ml/                 TensorFlow training, evaluation, calibration, gates, ONNX export
│   ├── src/medvision_ml/   data · models · training · evaluation · calibration · export
│   └── tests/              unit, pipeline and TensorFlow-vs-serving parity tests
├── backend/            FastAPI service on ONNX Runtime and NumPy (no TensorFlow)
│   └── app/                api · core · inference · services · schemas
├── frontend/           React + Vite + Tailwind web app, served by the backend in production
├── notebooks/          Kaggle notebook that runs the whole training pipeline
├── docs/               model card, architecture, API, datasets, deployment, project plan
├── Dockerfile          one image: built frontend, API and verified model
├── render.yaml         Render deployment (Blueprint)
└── Makefile            setup · check · run · evaluate · docker · smoke
```

## Quality gates

| Check | Where |
|---|---|
| ruff, mypy (strict), ESLint, Prettier | CI on every pull request |
| **266 automated tests**: 131 ml, 100 backend, 35 frontend; backend coverage 98% | CI on every pull request |
| Parity: ONNX vs Keras, NumPy heatmap vs TensorFlow Grad-CAM, served resize, out-of-distribution and noise scores | `ml` test suite |
| Docker build, then a container smoke test: ready, home page, predict, explain, non-skin and grainy images rejected, p95 latency | CI on every pull request |
| The same smoke test against the public URL | Daily and on demand ("Live site") |

## Limitations

- **High referral rate.** To catch 90% of malignant lesions, the flag fires on about half
  of the test images. ISIC 2019 is rich in cancers (37% of test images are malignant); in
  primary care, where cancers are rarer, a smaller share of flagged lesions would be
  malignant.
- **Moderate accuracy.** Top-1 recall is low for melanoma (0.43) and squamous cell
  carcinoma (0.17). That is why the referral answer relies on the combined malignancy
  probability, not on the top class.
- **Dermoscopy only.** Clinical and phone photos are rejected by design.
- **Eight lesion types only.** Rarer skin cancers (for example Merkel cell carcinoma or
  melanoma subtypes that are rare in the training data) may be missed;
  a "no flag" answer never rules out malignancy.
- **Heavy JPEG compression** (quality 10) still causes silent misses (25%).
- **Unequal recall** for women, ages 30–49 and the lower limbs; **unknown** performance
  on darker skin.
- One dataset, no external or prospective validation. See the
  [model card](docs/model_card.md) for details.

## Roadmap

| Item | Status |
|---|---|
| Leakage-free data pipeline, training and export | ✅ Done |
| Inference API, calibration, safety and input gates | ✅ Done |
| Web app, Docker image, CI and live deployment | ✅ Done |
| Robustness and subgroup evaluation, model card | ✅ Done |
| **Stronger model**: larger backbone, higher resolution, full fine-tuning, tracked in MLflow, on a Kaggle GPU; aims to keep recall at 0.90 with fewer referrals | Next: objective 8 |
| **Catch heavy JPEG compression** in the input gates | Planned: objective 4 |
| External validation on another dermoscopy dataset, including darker skin | Open: needs licensed data with skin-type labels |

A new model replaces the current one only if it keeps malignant recall at 0.90 or higher.
Among those, the highest macro F1 wins, then the lowest calibration error. Accuracy alone
never decides.

## Documentation

| Document | What it covers |
|---|---|
| [Model card](docs/model_card.md) | Intended use, data, per-class results, decision rules, robustness, subgroups, limitations |
| [Architecture](docs/architecture.md) | Components, request flow, model pipeline, key design decisions |
| [API reference](docs/api.md) | Endpoints, response fields, error codes, configuration |
| [Datasets](docs/DATASETS.md) | ISIC 2019, everyday-photo set, licences, attribution, training steps |
| [Deployment](docs/deploy.md) | Render setup and the live smoke test |
| [Project plan](docs/PROJECT_PLAN.md) | Problem, objective, scope, phases, conventions and the full decision log |

## Acknowledgements and licence

- **Training data:** the [ISIC 2019 Challenge](https://challenge.isic-archive.com/data/#2019)
  dataset (CC BY-NC 4.0), drawing on HAM10000 (Tschandl et al., 2018), the ISIC 2017
  challenge (Codella et al., 2017) and BCN20000 (Combalia et al., 2019). Full citations are
  in [docs/DATASETS.md](docs/DATASETS.md).
- **Everyday photos** for the out-of-distribution report: Open Images V4 validation set
  (CC BY 2.0), downloaded with attribution by `medvision-ml fetch-non-skin`.
- **Licences:** the source code is released under the [**MIT licence**](LICENSE). The model
  weights and gate statistics are released under **CC BY-NC 4.0**, because the training data is
  non-commercial.

<div align="center">
<br>
<sub>A skin-cancer referral check for dermoscopy images, from data to deployment.<br>
<b>Decision support, not a medical device.</b> The referral decision rests with the clinician.</sub>
</div>
