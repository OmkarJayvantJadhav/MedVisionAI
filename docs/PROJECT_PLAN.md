# MedVision AI — Project Plan

This is the single source of truth for building MedVision AI. Read it before
starting any task, and use it to settle decisions. If a situation is not
covered here, apply the **decision rules** (§2), make the call, and record it
in the **decision log** (§12).

---

## 1. Goal

Ship an end-to-end, explainable skin-lesion classification product:

> Upload a skin photo → get the top-3 predicted conditions with **calibrated**
> confidence, a **Grad-CAM heatmap** showing what the model looked at, a
> **safety status** (ok / low confidence / rejected), and a plain-language
> summary that never claims to be a diagnosis.

The project must demonstrate production engineering, not just model training:
a tested API, a usable UI, one-command local run, CI, a public live demo,
reproducible training, and responsible-AI evaluation.

**Out of scope:** eye and oral disease models (the prototype had them; they are
dropped), user accounts, storing patient data, mobile apps.

---

## 2. Decision rules (apply in this order)

1. **Patient safety and correctness first.** Never ship a change that makes
   the system more confident or less honest. When in doubt, reject or flag
   low confidence instead of guessing.
2. **A working end-to-end demo beats a better model.** Do not tune the model
   while the API, UI, or deployment is broken or missing.
3. **Tested and reproducible beats clever.** Every feature lands with tests.
   Every number in the README must be reproducible by a script in the repo.
4. **Simple, boring, free.** CPU-only inference, free hosting tiers, no paid
   services required to run the project. Prefer the standard library and
   well-known tools over new dependencies.
5. **The serving path never imports TensorFlow.** TensorFlow lives only in
   `ml/`. The backend runs on ONNX Runtime + NumPy.
6. **Small, reviewable changes.** One concern per commit/PR. Don't mix
   refactors with behavior changes.
7. **When two options are close, pick the one that is easier to explain in an
   interview** and write down why.

---

## 3. Starting point (facts from the prototype)

Migration source: the prototype repository
`github.com/Adiitya09/MedvisionAi`, folder `ML-Model/` (~11k lines Python,
TensorFlow/Keras).

**What exists and is reusable**

| Prototype path | Reuse as |
|---|---|
| `config.py`, `config/config.yaml` | `ml/` config (rewrite: drop eye/oral, drop hardcoded `E:\...` path, no side effects on import) |
| `utils/model_builder.py` | model construction (EfficientNetB0 / MobileNetV2 / ResNet50, frozen backbone + GAP + Dropout + Dense softmax) |
| `utils/dataset_loader.py`, `callbacks.py`, `metrics.py`, `visualization.py` | training utilities |
| `train.py` | two-stage training (frozen head, then fine-tune) |
| `evaluation/evaluate.py`, `error_analysis.py` | evaluation and reports |
| `explainability/explainability.py` | TensorFlow Grad-CAM — **reference implementation** for parity tests |
| `optimization/optimize_model.py` | ONNX / TFLite export (via tf2onnx) |
| `scripts/clean_skin_eye_datasets.py`, `split_medical_dataset.py` | data cleaning and splitting (keep skin parts only) |
| `models/skin_model.keras`, `models/skin_class_names.json` | current trained model (do **not** commit to git; publish as a release asset — see §5) |

**Do not migrate:** `train_cleaned_oral.py`, eye/oral code paths and outputs,
the bulk of `outputs/`, the stray `git status` file, the empty
`Backend/` and `Frontend/` folders.

**Model facts**

- Input: 224×224×3 RGB, float32, **raw 0–255 pixels**. Preprocessing is
  inside the model; do not normalize again outside it.
- Architecture: augmentation layers (inactive at inference) → backbone
  preprocessing → EfficientNetB0 (ImageNet) → GlobalAveragePooling → Dropout
  (0.3) → Dense(8, softmax).
- Classes (index order is fixed, from `skin_class_names.json`):
  `0 Acne, 1 Actinic Keratosis, 2 Basal Cell Carcinoma, 3 Dermatofibroma,
  4 Melanocytic Nevus (Nevus), 5 Melanoma, 6 Seborrheic Keratosis,
  7 Squamous Cell Carcinoma`.
- Malignant / pre-malignant (safety-critical) classes: **Melanoma, BCC, SCC,
  Actinic Keratosis**.

**Prototype baseline test results** (prototype dataset with Acne, undocumented
source; kept for reference only — see D21; 2,280 test images, imbalance ≈ 25:1)

| Metric | Value |
|---|---|
| Accuracy | 76.32% |
| Macro F1 | 0.6851 |
| Weighted F1 | 0.7567 |
| Multi-class ROC-AUC | 0.9608 |
| Melanoma recall | **0.599** (main weakness) |
| SCC recall | 0.391 |
| Dermatofibroma recall | 0.343 (35 test images only) |

The prototype also tried a sqrt-moderated class-weight model: accuracy 73.86%,
macro F1 0.6617, but better recall on SCC (0.594) and AK (0.808). Neither
model is clearly better. Choose by the safety-weighted metric in §8.

**Runtime benchmarks** (CPU, single image): Keras ≈ 284 ms, ONNX ≈ 5.5 ms
(identical predictions), TFLite FP16 ≈ 12 ms (≈ identical), TFLite INT8 ≈ 8 ms
but −9.5 pts accuracy (**not acceptable**).

**Known risks to verify early**

- The prototype mentions a "cleaned & deduplicated" split was introduced
  later. Check for **train/test leakage** (duplicate or near-duplicate images,
  same lesion in several splits) before trusting any number.
- Dataset provenance and license are undocumented. Record them before
  publishing any data-derived artifacts.

---

## 4. Target architecture

```
                ┌──────────────────────────────────────────────┐
  Browser  ───▶ │  Single container (FastAPI)                  │
  (React UI)    │   • serves built frontend (static files)     │
                │   • /api/v1/predict  /api/v1/explain  /health│
                │   • ONNX Runtime + NumPy (no TensorFlow)     │
                │   • calibration + OOD gate + CAM in NumPy    │
                │   • optional LLM summary (pluggable, off by  │
                │     default, template fallback)              │
                └──────────────────────────────────────────────┘
                             ▲ model artifacts (downloaded at build,
                             │ checksum-verified)
  ml/ (TensorFlow, offline) ─┘  train → evaluate → calibrate → export
```

### Key design: Grad-CAM without TensorFlow

The head is `GAP → Dropout → Dense`. For this architecture the Grad-CAM
channel weights for class *c* equal the Dense kernel column `W[:, c]`
(divided by the constant H·W). So Grad-CAM reduces to **CAM**:
`cam = ReLU(Σ_k W[k, c] · A_k)`, where `A` is the last conv feature map.

- Export the ONNX model with **two outputs**: `probabilities` and
  `feature_maps` (last conv activation, 7×7×1280 for EfficientNetB0).
- Ship `head_weights.npz` (Dense kernel + bias) next to it.
- Compute the heatmap in NumPy, upsample to the image size, and overlay it.
- **Parity test:** the NumPy CAM must match the TensorFlow Grad-CAM reference
  on fixed images (cosine similarity ≥ 0.99 after normalization). If it ever
  doesn't, that test fails the build.

If the architecture changes (for example an extra Dense layer in the head),
this shortcut no longer holds. Then export a gradient-free alternative or
revisit this decision in §12.

---

## 5. Tech stack decisions

| Area | Decision | Why / rejected alternatives |
|---|---|---|
| Language | Python 3.11 | TF 2.15+ and ORT support; 3.12 is fine for backend only |
| Packaging | `uv` + `pyproject.toml` per package, lockfile committed | fast, reproducible; rejected bare `requirements.txt` |
| Training | TensorFlow/Keras (existing) | reuse working pipeline; no rewrite to PyTorch |
| Serving runtime | ONNX Runtime (CPU) | 50× faster than Keras in benchmarks, small image, no TF |
| API | FastAPI + Pydantic v2 + Uvicorn | typed contracts, auto OpenAPI docs |
| Image I/O | Pillow + NumPy | must replicate `tf.keras.utils.load_img` resize (bilinear) — covered by a parity test |
| Frontend | React (JavaScript, JSX) + Vite + Tailwind | industry standard; plain JavaScript by owner's choice (D31); rejected Streamlit/Gradio as primary UI |
| Lint / format / types | ruff (lint + format), mypy (strict on `backend/` and `ml/src`), ESLint + Prettier for the frontend | |
| Tests | pytest (+ httpx TestClient), Vitest for frontend | |
| Pre-commit | `pre-commit` running ruff, mypy, end-of-file fixers | |
| CI | GitHub Actions | free for public repos |
| Container | one multi-stage Dockerfile (node build → python slim runtime) | one image = one deploy target |
| Hosting | Hugging Face Spaces (Docker SDK) for the live demo | free CPU; rejected paid clouds |
| Model artifacts | GitHub Release assets (`model.onnx`, `head_weights.npz`, `model_meta.json`) with SHA-256 in `model_meta.json`; mirror to the Hugging Face Hub with a model card | keeps git small; rejected committing binaries / Git LFS quota |
| Experiment tracking | MLflow, local file store (`ml/mlruns/`, gitignored) | already used by the prototype |
| Data versioning | DVC, optional, only once a remote is chosen | the dataset is not public yet |
| LLM (optional feature) | provider-agnostic interface, configured by env vars, **disabled by default** | the demo must work with no API keys |

---

## 6. Repository layout

```
medvisionai/
├── README.md                  # case-study style overview + live demo link
├── docs/
│   ├── PROJECT_PLAN.md        # this file
│   ├── architecture.md        # diagrams, data flow (Phase 9)
│   ├── model_card.md          # Phase 6
│   └── api.md                 # generated / summarized from OpenAPI
├── ml/                        # TensorFlow training & research (offline only)
│   ├── pyproject.toml
│   ├── configs/skin.yaml
│   ├── src/medvision_ml/
│   │   ├── config.py
│   │   ├── data/              # loading, cleaning, splitting, leakage checks
│   │   ├── models/            # builder, backbones
│   │   ├── training/          # train loop, callbacks, class weights
│   │   ├── evaluation/        # metrics, reports, error analysis, fairness
│   │   ├── explainability/    # TF Grad-CAM reference
│   │   ├── calibration/       # temperature scaling, OOD threshold fitting
│   │   └── export/            # ONNX export (2 outputs), head weights, meta
│   ├── scripts/               # thin CLIs calling src/
│   └── tests/
├── backend/                   # FastAPI service (NO TensorFlow)
│   ├── pyproject.toml
│   ├── app/
│   │   ├── main.py            # app factory, static frontend mount
│   │   ├── api/v1/            # routers: predict, explain, health, meta
│   │   ├── core/              # settings (pydantic-settings), logging, errors
│   │   ├── inference/         # preprocessing, onnx session, cam, calibration, ood, download
│   │   ├── services/          # summary (template + optional LLM), rag
│   │   └── schemas/           # pydantic request/response models
│   └── tests/
├── frontend/                  # React (JavaScript) + Vite + Tailwind
├── models/                    # downloaded artifacts (gitignored, has .gitkeep)
├── Dockerfile
├── docker-compose.yml
├── Makefile                   # make setup / test / lint / run / docker
└── .github/workflows/ci.yml
```

Rules: no code at the repo root besides tooling; `scripts/` files are thin
wrappers (the logic lives in `src/` or `app/` so it is testable); generated
outputs go to gitignored folders.

---

## 7. API contract (v1)

All endpoints live under `/api/v1`. Errors use a consistent body:
`{"error": {"code": "string", "message": "string"}}`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness: `{"status":"ok"}` |
| GET | `/ready` | readiness: model loaded, checksum verified, model version |
| GET | `/meta` | model version, class list, thresholds, disclaimer text |
| POST | `/predict` | multipart `file`; returns prediction (below) |
| POST | `/explain` | multipart `file`; returns prediction + heatmap PNG (base64) and overlay |

`/predict` response:

```json
{
  "status": "ok | low_confidence | rejected",
  "predicted_class": "Melanoma",
  "confidence": 0.71,
  "top_k": [{"class_name": "Melanoma", "probability": 0.71}, "..."],
  "risk_flag": "malignant_suspected | benign | uncertain",
  "model_version": "skin-effb0-v1.0.0",
  "calibrated": true,
  "summary": "plain-language text, always with the disclaimer",
  "disclaimer": "Not a medical diagnosis. ...",
  "latency_ms": 7.4
}
```

Input limits: JPEG/PNG/WebP only, ≤ 8 MB, longest side ≤ 4096 px, verified
by decoding the image, not trusting the extension. Uploaded images are
**never written to disk or logged**.

---

## 8. Safety and responsible-AI rules (non-negotiable)

1. Every response and UI screen shows the disclaimer: educational and
   screening research only, not a diagnosis, consult a dermatologist.
2. **Calibration:** fit temperature scaling on the validation set. Ship `T`
   in `model_meta.json`. Report ECE and a reliability diagram before and
   after. Target ECE ≤ 0.05.
3. **Rejection gate (status field):**
   - `rejected`: the input is likely not a skin image (OOD score past the
     threshold) or is unusable (too dark or blurry, measured by simple
     image-quality checks).
   - `low_confidence`: calibrated max probability < threshold `τ`.
   - Fit thresholds on validation data plus a small set of non-skin images
     (stored as a test fixture with a license note). Start with max-softmax
     and the energy score; keep whichever separates better (AUROC).
4. **Malignancy safety:** if the summed calibrated probability of
   {Melanoma, BCC, SCC, AK} ≥ `τ_malignant`, set `risk_flag =
   malignant_suspected`, even when the top class is benign. Pick
   `τ_malignant` so malignant-group recall on validation is ≥ 0.90, and
   report the resulting precision honestly.
5. **Model selection metric:** among candidate models, choose by
   (a) malignant-group recall at the operating point, then (b) macro F1,
   then (c) ECE. Do not choose by accuracy alone.
6. **Fairness and robustness reporting** (Phase 6): per-class metrics,
   performance under blur, brightness, JPEG compression and phone-camera
   style augmentations; per-skin-tone metrics if Fitzpatrick labels or
   another proxy are available. Otherwise state the gap explicitly in the
   model card.
7. The LLM summary may only restate model outputs and retrieved reference
   text. It must not invent a diagnosis, treatment, or certainty. If the LLM
   is unavailable or its output fails validation, use the template summary.
8. Never present a number in the README that a script in the repo cannot
   reproduce.

---

## 9. Phases and definition of done

Work strictly in order unless a later phase blocks nothing. Each phase ends
with green CI and an updated README section.

### Phase 0 — Bootstrap ✅ (this commit)
- [x] Project plan, README stub.

### Phase 1 — Repo tooling
- [x] `.gitignore`, `.editorconfig`, `pre-commit`, `Makefile`, root README
      skeleton.
- [ ] MIT license — blocked on §13 question 2 (confirm with all original
      contributors first).
- [x] CI workflow: ruff, mypy, pytest for `ml/` and `backend/` (initially
      passing on placeholder tests), frontend lint/build placeholder.
- **Done when:** `make lint test` passes locally and in CI.

### Phase 2 — Migrate the ML pipeline (skin only)
- [x] Port prototype modules into `ml/src/medvision_ml/` per §3; remove
      eye/oral paths; config from YAML + env vars; no directory creation or
      other side effects at import time.
- [x] Data leakage check (`medvision-ml check-leakage`: MD5 + dHash
      duplicates across splits) with a JSON report, and a duplicate-aware
      `medvision-ml split`.
- [x] Dataset chosen (D21): `medvision-ml prepare-isic2019` converts the
      ISIC 2019 release and records lesion groups; `split` and
      `check-leakage` respect them (D22).
- [x] Run prepare → split → check-leakage on the downloaded images (9.8 GB):
      48 duplicate copies and 4 conflicting-label images removed, 25,279 images
      split 80/10/10 per class, no leakage (after the D37 fix).
- [x] Unit tests: config parsing, class-weight computation, model builder
      output shapes (`weights=None`), metrics functions, Grad-CAM parity, and
      an end-to-end train → evaluate → Grad-CAM smoke test.
- **Done when:** `ml` tests pass in CI without the dataset (dataset-dependent
  tests are marked `@pytest.mark.data` and skipped in CI).

### Phase 3 — Export and inference core
- [x] Export ONNX with outputs `probabilities` and `feature_maps`, plus
      `head_weights.npz` and `model_meta.json` (version, classes, input spec,
      temperature, thresholds, SHA-256 checksums): `medvision-ml export`.
- [x] `backend/app/inference/`: preprocessing matching the training resize;
      ORT session (loaded once); NumPy CAM; temperature scaling; OOD and
      quality gate.
- [x] Parity tests (`ml/tests/test_export_parity.py`): ONNX vs Keras
      probabilities (≤ 1e-4), serving CAM vs TF Grad-CAM (cosine ≥ 0.99),
      serving preprocessing vs `image_dataset_from_directory` (PNG mean abs
      diff ≤ 1e-3; JPEG ≤ 2.0 because of decoder differences, see D15).
- [x] Checksum-verified downloader (`python -m app.inference.download`,
      `make download-model`).
- [x] Kaggle notebook (`notebooks/train_kaggle.ipynb`, D39) that downloads,
      prepares, splits, leakage-checks, trains, evaluates and exports; dry-run
      end to end on a 320-image ISIC 2019 subset.
- [x] Train on the ISIC 2019 splits on Kaggle, then publish release
      `model-v2.0.0` under CC BY-NC 4.0 from the exported, calibrated files
      (re-exported after the D40 fix). `make download-model` fetches and
      verifies it.
- **Done when:** a CLI `python -m app.inference.cli image.jpg` prints a
  prediction with no TensorFlow installed. ✅ Verified with the exported
  prototype model (16 ms per prediction, 74 ms with explanation, CPU).

### Phase 4 — Backend API
- [x] Endpoints per §7, settings via `pydantic-settings`, structured JSON
      logging (request id, latency, status; never image content), upload
      validation, CORS restricted to configured origins, simple per-IP rate
      limit. Reference: `docs/api.md`.
- [x] Tests: happy path, invalid file types, oversized files, corrupted
      image, rejected images, low-confidence path, malignancy flag logic,
      error body shape, rate limiting, CORS, missing model, hidden internal
      errors, access-log content. Uses the in-memory fake ONNX model; the
      real-model checks live in the `ml` parity tests.
- **Done when:** OpenAPI docs render at `/docs`, coverage of `app/` ≥ 85%.
  ✅ 98% coverage; verified with uvicorn on the exported prototype model
  (`/predict` ≈ 46 ms, `/explain` ≈ 92 ms end to end).

### Phase 5 — Frontend
- [x] Upload (drag and drop + camera capture on mobile), preview, results
      card with top-3 bars, status and risk badges, heatmap/overlay toggle
      with an opacity slider, disclaimer always visible, loading and error
      states, responsive layout, accessible (labels, contrast, keyboard).
- [x] API client in plain JavaScript; frontend tests check their mock
      responses against the committed OpenAPI schema
      (`frontend/src/api/openapi.json`), and a backend test fails if that
      schema is stale.
- [x] Vitest tests for the results rendering, status badge logic, upload
      panel, API client and the full upload → result flow.
- **Done when:** the production build is served by FastAPI from one
  container. ✅ Verified locally: FastAPI serves `frontend/dist` with
  `MEDVISION_STATIC_DIR`, and a headless-Chromium run (desktop and mobile)
  completed upload → result with no console errors. The container itself
  is Phase 7.

### Phase 6 — Responsible-AI evaluation
- [x] Calibration (temperature scaling, ECE, reliability diagrams before and
      after): `medvision-ml calibrate` (D41).
- [x] Threshold fitting for `τ` and `τ_malignant` on validation, reported on
      test; `export --calibration` ships them (D41).
- [x] OOD gate: Mahalanobis distance of the pooled features, threshold from
      validation, reported on 497 CC BY 2.0 Open Images photos (D45); shipped
      in release `model-v2.1.0`; noise check added in `model-v2.2.0` (D47).
- [x] Robustness suite (blur, brightness, JPEG, noise, rotation, phone-like):
      `medvision-ml robustness` (D46).
- [x] Fairness analysis by sex, age and body site (`medvision-ml subgroups`);
      skin type is not recorded in ISIC 2019, so that gap is documented (D46).
- [x] `docs/model_card.md`: intended use, data, metrics, limitations,
      ethical considerations.
- **Done when:** all §8 artifacts are generated by `make evaluate` and
  linked from the README. ✅ The model card quotes only `make evaluate`
  output and is linked from the README.

### Phase 7 — Ship
- [x] Multi-stage Dockerfile (non-root user, health check, image < 1 GB
      target), `docker-compose.yml`: 829 MB, model checksum-verified at build.
- [x] CI builds the image and smoke-tests the container (`/ready`, home page,
      `/predict`, `/explain`, p95 latency) (D43).
- [x] `render.yaml`: Render deploys `main` after CI passes; the **Live site**
      workflow smoke-tests the public URL daily and on demand (D44).
- [x] Owner creates the Render service from the Blueprint and sets `LIVE_URL`
      (docs/deploy.md).
- **Done when:** the public demo link works and is at the top of the README.

### Phase 8 — Explanations with an LLM (optional, after Phase 7)
- [ ] `services/summary.py`: interface `SummaryProvider` with
      `TemplateSummaryProvider` (default) and `LLMSummaryProvider`
      (env-configured, timeout ≤ 10 s, output validated by schema and
      banned-phrase checks, falls back to the template).
- [ ] Small RAG: curated, license-checked condition descriptions in
      `backend/app/knowledge/*.md`; embed with a small local embedding model;
      cosine top-k retrieval; return citations with the summary.
- [ ] Evaluation set of about 30 cases checking faithfulness (no claims
      beyond the model output and sources) and that the disclaimer is
      present.
- **Done when:** the summary works with no key configured (template) and
  with a key (LLM + citations), and the eval script passes.

### Phase 9 — Model improvements (only after Phases 1–7)
Each experiment is logged in MLflow, compared using the §8 selection metric,
and summarized in an ablation table.
- [ ] Re-baseline on leakage-free splits.
- [ ] Backbones: EfficientNetV2-S, ConvNeXt-Tiny (keep GAP + Dense head so
      CAM still works).
- [ ] Higher resolution (288 / 320), focal loss vs sqrt-moderated class
      weights, stronger augmentation, test-time augmentation (latency budget
      ≤ 100 ms on CPU).
- **Targets:** macro F1 ≥ 0.72, melanoma recall ≥ 0.70 at the top-1
  decision, malignant-group recall ≥ 0.90 at `τ_malignant`, ECE ≤ 0.05.
  These are goals, not claims. Report what is actually achieved.

### Phase 10 — Presentation
- [ ] README as a case study: demo link + GIF, problem, architecture
      diagram, results table, safety design, trade-offs (ONNX vs INT8,
      CAM shortcut, calibration), limitations, one-command run.
- [ ] `docs/architecture.md`; a short write-up of the three most interesting
      engineering decisions.

---

## 10. Conventions

**Code**
- Type hints everywhere; mypy strict for `backend/app` and `ml/src`.
- Google-style docstrings on public functions; no commented-out code.
- No global state with side effects at import time; configuration objects are
  created explicitly and injected (FastAPI dependencies for the backend).
- Raise specific exceptions; the API maps them to error codes in one place.
- Log with `logging` (JSON in the backend); never `print` outside CLIs.
- Constants such as class names come from `model_meta.json`, never
  hardcoded in several places.

**Tests**
- Every bug fix comes with a regression test.
- Markers: `data` (needs the dataset), `model` (needs real model
  artifacts), `slow`. The default CI run excludes `data`; `model` runs in a
  separate CI job after downloading artifacts.

**Git**
- Branches: `feat/...`, `fix/...`, `chore/...`, `docs/...`; short-lived; merge
  to `main` with squash.
- Commit messages: Conventional Commits (`feat(api): add /explain
  endpoint`), imperative mood, body explains *why* when not obvious.
- Commit messages contain only the change description: **no co-author
  trailers, tool attributions, or generated-by footers.**
- Tracked files describe the project only. They do not reference the editors,
  assistants, or other tooling used to write them. Tool-specific config
  folders and files stay untracked.
- Never commit datasets, model binaries, `.env` files, MLflow runs, or
  generated outputs (except small curated figures under `docs/assets/`).

**Versioning**
- Model artifacts: `model-vMAJOR.MINOR.PATCH` release tags; any change to
  classes or input spec is a MAJOR bump.
- API: path-versioned (`/api/v1`); breaking changes need `/api/v2`.

---

## 11. Quality gates (CI must enforce)

- ruff, mypy, ESLint and Prettier clean.
- pytest (backend coverage ≥ 85%), Vitest.
- Parity tests (§9 Phase 3) in the `model` job.
- Docker image builds; container `/ready` returns 200 in a smoke step.
- p95 `/predict` latency ≤ 150 ms on the CI runner (single image, warm) —
  warning only, not failing, at first.

---

## 12. Decision log

Append new entries; never rewrite old ones (add a superseding entry instead).

| # | Date | Decision | Reason |
|---|---|---|---|
| D1 | 2026-10-16 | Skin domain only; eye/oral dropped | focus; their results were weak (oral 49% acc) |
| D2 | 2026-10-16 | Serve with ONNX Runtime, no TF in the backend | 5.5 ms vs 284 ms, smaller image, same predictions |
| D3 | 2026-10-16 | Grad-CAM computed as CAM in NumPy from ONNX feature maps + head weights | mathematically equivalent for a GAP+Dense head; removes the TF dependency; guarded by a parity test |
| D4 | 2026-10-16 | Reject TFLite INT8 | −9.5 pts accuracy, −0.15 macro F1 |
| D5 | 2026-10-16 | One container serves API + static frontend, hosted on HF Spaces | free, single deploy target |
| D6 | 2026-10-16 | Model binaries in GitHub Releases (+ HF Hub mirror), not in git | repo size; versioned, checksum-verified artifacts |
| D7 | 2026-10-16 | Select models by malignant-group recall → macro F1 → ECE | patient safety over headline accuracy |
| D8 | 2026-10-16 | LLM summary optional, provider-agnostic, off by default, template fallback | demo must run without keys; safety |
| D9 | 2026-10-16 | `ml/` and `backend/` are independent uv projects with separate lockfiles (no uv workspace); one shared root `ruff.toml` | the backend image must not resolve TensorFlow; `ml/` is pinned to Python 3.11 for TF while the backend is not |
| D10 | 2026-10-16 | Supported backbones: EfficientNetB0 (default), EfficientNetV2-S, ConvNeXt-Tiny, MobileNetV2; ResNet50 dropped. No `Lambda` layers: preprocessing is built into the backbone or done with `Rescaling` | `Lambda` layers only load in Keras unsafe mode (arbitrary code); ResNet50's channel-flip preprocessing would need one |
| D11 | 2026-10-16 | Head is `pool → dropout → logits (Dense, linear) → probabilities (softmax)`; Grad-CAM targets the backbone output and differentiates the **logit** | exposes logits for temperature scaling; makes NumPy CAM exact. The prototype differentiated the softmax output at the last `Conv2D` (before BN + activation), which has no closed form. Verified on the prototype model: NumPy CAM vs TF Grad-CAM cosine 1.0, max abs diff 2.4e-7 |
| D12 | 2026-10-16 | One `medvision-ml` console entry point with subcommands instead of `ml/scripts/*.py` | one tested CLI, no `sys.path` hacks |
| D13 | 2026-10-16 | Splits are built per class from near-duplicate groups (dHash Hamming ≤ 4, union-find), largest-remainder allocation, SHA-256-derived per-class seed | copies of one image can never straddle splits; small classes still get validation/test images; seeds are stable across Python versions |
| D14 | 2026-10-16 | Mixed precision only when a GPU is present; default class weighting `sqrt` | mixed precision slows CPU training; sqrt weights were the prototype's best trade-off |
| D15 | 2026-10-16 | Serving decodes with Pillow (EXIF orientation applied) and resizes with a NumPy re-implementation of `tf.image.resize` bilinear (half-pixel centres, no antialias) | matches the training pipeline exactly for lossless input. The prototype's inference used `load_img` (nearest-neighbour), which did not match training. JPEG still differs by ~1 grey level because TF uses the fast integer DCT |
| D16 | 2026-10-16 | Export freezes variables to constants, then converts with tf2onnx (opset 17); the backend recomputes logits from `feature_maps` and the head weights instead of using the `probabilities` output | without freezing, EfficientNet's normalisation statistics become graph inputs; recomputing logits gives temperature scaling and the energy score from one forward pass |
| D17 | 2026-10-16 | Provisional thresholds until Phase 6: low confidence < 0.50, malignant-group probability ≥ 0.30, brightness 20–235, sharpness ≥ 5, OOD gate off; responses report `thresholds_fitted: false` | errs towards flagging; the sharpness bound is low because dermoscopy images are naturally smooth |
| D18 | 2026-10-16 | Model download logic lives in `backend/app/inference/download.py` (not `scripts/`); files are verified and moved atomically, metadata written last | testable, reusable by the Docker build, a partial download never looks complete |
| D19 | 2026-10-16 | `tf2onnx`, `onnx`, `onnxruntime` are regular `ml` dependencies, not an extra | `uv run` prunes extras; the export and parity tests must always run |
| D20 | 2026-10-16 | Heatmap and overlay PNGs use `compress_level=1` | `optimize=True` took ~400 ms per explanation; now ~60 ms |
| D21 | 2026-10-16 | Training data is the ISIC 2019 training release (25,331 dermoscopic images, CC BY-NC). Classes become ISIC's 8: AK, BCC, Benign Keratosis, DF, Melanoma, Melanocytic Nevus, SCC, Vascular Lesion. Acne is dropped. Model weights are published under CC BY-NC 4.0 with attribution | documented, expert-labelled and redistributable for non-commercial use; acne has no comparable free dataset and is a different imaging modality (shortcut-learning risk). See `docs/DATASETS.md` |
| D22 | 2026-10-16 | Splits are grouped by ISIC `lesion_id` merged with near-duplicate groups; `check-leakage` also fails when a lesion spans splits (`groups.csv`) | 16,459 of 25,331 images belong to multi-image lesions; per-image splitting would leak lesions into the test set |
| D23 | 2026-10-16 | The prototype model (with Acne) is used only for development until an ISIC 2019 model is trained; the new model is a MAJOR version (`skin-effb0-v2.0.0`) | its classes and data provenance differ; release rule in §10 |
| D24 | 2026-10-16 | Train on Google Colab (free T4) through a notebook that only calls the `medvision-ml` CLI; every step fails fast; results are copied to Google Drive | free GPU; the notebook adds no logic of its own, so what runs there is what CI tests |
| D25 | 2026-10-16 | `data.cache_dir` caches decoded images on disk | in-memory caching of ISIC 2019 needs ~12 GB, more than a free Colab VM |
| D26 | 2026-10-16 | Splitting gives each lesion group (largest first) to the split furthest below its target | the cumulative-boundary method could leave a small class with no validation or test images when a multi-photo lesion overshot a quota (found in the notebook dry run) |
| D27 | 2026-10-16 | App factory (`uvicorn app.main:create_app --factory`); the model loads once in the lifespan hook; if it fails the service still starts, `/health` is 200 and `/ready`, `/predict`, `/meta` return 503 | no side effects on import; orchestrators can tell "process up" from "model ready" |
| D28 | 2026-10-16 | In-memory sliding-window rate limit (30 requests/min per client on `/api/`); `X-Forwarded-For` only when `MEDVISION_TRUST_FORWARDED_FOR=true` | enough for a single-instance demo without Redis; trusting the header by default would let clients bypass the limit |
| D29 | 2026-10-16 | Error status mapping: 400 invalid image / unknown class, 413 too large, 415 wrong type, 422 validation, 429 rate limit, 503 model, 500 internal (details logged only) | one handler module; stable `code` values for the frontend |
| D30 | 2026-10-16 | Summaries come from a deterministic `TemplateSummaryProvider` behind a `SummaryProvider` protocol | safe default with no external calls; Phase 8 adds an LLM provider behind the same interface |
| D31 | 2026-10-16 | Frontend in plain JavaScript (JSX), no TypeScript | owner's decision. API contract safety comes from a committed OpenAPI snapshot instead: the backend test fails when it is stale, and frontend tests validate mock responses against it |
| D32 | 2026-10-16 | The UI makes one `/explain` call per analysis and layers the heatmap PNG over the user's own image in the browser; opacity is a CSS change | one round trip (~90 ms) gives result and heatmap; the slider needs no extra requests. Both images share the aspect ratio (EXIF orientation is applied on both sides) |
| D33 | 2026-10-16 | The footer links to the dataset page instead of naming the training data | the served model can be the prototype or the ISIC model; the UI must not state something about the model that may be untrue |
| D34 | 2026-10-16 | Before splitting, identical copies of an image are reduced to one; when copies disagree on the diagnosis all of them are dropped (listed in `outputs/split_removed.json`). Duplicate groups that mix labels are stratified under their most common label so they stay in one split | the first real ISIC 2019 split had 2 exact and 24 near-duplicate pairs across splits, all from images filed under two diagnoses (26 label conflicts); a conflicting label is unknown ground truth |
| D35 | 2026-10-17 | The input pipeline loads single images, caches them, shuffles images (buffer 1,024 ≈ 600 MB) and only then batches | shuffling batches held ~1,000 batches (≈12 GB) in RAM and crashed the first ISIC run on a free Colab VM; it also kept the same images together in every epoch |
| D36 | 2026-10-17 | `load_model` rebuilds mixed-precision models with float32 layers (weights copied unchanged), so evaluation, Grad-CAM and export all compute in float32 | the first ISIC model was trained with `mixed_float16`; Keras then computed export reference probabilities in float16 and the ONNX parity check failed (3.5e-3 > 1e-4). Served ONNX runs in float32, so every offline number must too |
| D37 | 2026-10-17 | Near-duplicates use a 256-bit difference hash of the central 60% of the image (threshold 16 bits); `split` stops if a near-duplicate group exceeds 20 images. Supersedes the hash in D13. The first ISIC model's test numbers are void | the 64-bit whole-frame hash mostly encoded the dark circular border many dermoscopy images share: at ≤ 4 bits it chained 3,676 unrelated images of all 8 classes into one "duplicate" group, which the splitter put wholly into training, leaving validation and test 71% nevus with half the expected melanomas. On ISIC 2019 the new hash's matches up to 18 bits were all copies of the same lesion photo (checked visually) and unrelated lesions start at 19; the largest group is now 3 images and every class is split 80/10/10 |
| D38 | 2026-10-17 | The Colab notebook is removed; training, evaluation and export run directly through the `medvision-ml` CLI on any machine. Supersedes D24 | the notebook needed someone to keep a browser tab active for hours, and free sessions disconnected twice mid-run; the CLI is the tested path anyway, and the notebook only wrapped it |
| D39 | 2026-10-17 | Train on Kaggle's free GPU (T4) through `notebooks/train_kaggle.ipynb`, run as a background commit; the notebook only calls the `medvision-ml` CLI, fails fast, deletes the downloaded images once split and writes results to `/kaggle/working` (the Output tab). Builds on D38 | a committed Kaggle run keeps going with the browser closed (up to 12 h, about 30 GPU hours a week free), which removes the babysitting that made Colab unreliable; the CLI stays the only tested logic |
| D40 | 2026-10-17 | `to_float32` loads the weights into the float32 rebuild through Keras's own weight loader (`save_weights` / `load_weights`) instead of `set_weights`. Corrects D36; numbers from evaluations and exports of mixed-precision models made before this fix are void | `set_weights` skipped the Normalization layer's load hook, so EfficientNet's input normalisation silently became the identity in every float32 rebuild. The first Kaggle model scored 0.695 validation accuracy in training but 0.646 after reload (test 0.636); with the fix the float32 rebuild matches the trained model to 1.5e-3 with identical top-1 predictions |
| D41 | 2026-10-17 | `medvision-ml calibrate` fits on Validation and reports on Test: temperature by minimising NLL (golden-section search on log T, NumPy only); `τ_malignant` = highest summed malignant probability reaching 0.90 malignant recall; `τ` = lowest top-1 probability above which predictions are ≥ 85% correct, or 1.0 (everything low confidence) when no threshold reaches it. Targets live in `evaluation.target_*`; `export --calibration` writes them into `model_meta.json` with `fitted: true`. The OOD energy gate stays off until a licensed non-skin image set exists | §8.2–8.4. 85% makes an "ok" answer right far more often than the model overall (71%); falling back to 1.0 never lets a weak model look confident. SciPy is only a transitive dependency, so the 1-D search is written out instead of imported. First ISIC model: T = 1.08, test ECE 0.026, `τ_malignant` 0.193 → test malignant recall 0.904 at precision 0.681 (49% of images flagged), `τ` 0.625 → 59% of test images answered at 86.9% accuracy |
| D42 | 2026-10-17 | Frontend redesign: the verdict (malignancy flag first, then unsure, then benign; rejected photos say what to fix) leads the result, followed by the closest match, top-3 bars, a combined-malignancy gauge with the fitted threshold marked, the heatmap and collapsible technical details. A photo is analysed as soon as it is chosen (file, drop, paste or camera); three ISIC 2019 test-split images are offered as samples; fonts are self-hosted (`@fontsource`); model facts are read from `/meta`, never hard-coded | one fewer click and a sample to try lower the barrier for a demo; leading with what to do next is the safety-relevant part of the answer; self-hosting fonts keeps a medical tool free of third-party requests; reading facts from `/meta` keeps the page true when the model changes (D33). Sample images are small (≤ 45 KB) CC BY-NC test images, credited in `docs/DATASETS.md` |
| D43 | 2026-10-17 | One multi-stage image (`node:24-slim` build → `python:3.12-slim` runtime with uv-synced backend dependencies), model downloaded and checksum-verified at build time from the GitHub release, non-root uid 1000, port 7860, Docker health check on `/health`. CI builds it and runs `.github/scripts/smoke_test.py` against the container. `deploy.yml` uploads only tracked `backend/` and `frontend/` files plus a Space README with `huggingface_hub` after CI passes on `main`, waits for the Space to report a rebuild then `RUNNING`, and smoke-tests the live URL. On the Space the rate limit is 60 requests/min and `X-Forwarded-For` stays untrusted | uid 1000 and port 7860 are Hugging Face's defaults; `upload_folder` handles binary files (sample images) that a plain git push to a Space rejects; waiting for a rebuild stops the smoke test passing against the previous build. The app trusts the first `X-Forwarded-For` hop, which a client can forge, so behind the Space proxy the limit is a shared ceiling that protects the free CPU instead of a per-visitor limit. Measured locally: 829 MB image, `/predict` p95 28 ms |
| D44 | 2026-10-17 | Host the live demo on Render's free web-service tier from `render.yaml` (Docker runtime, deploy after CI checks pass, health check `/api/v1/health`, `MEDVISION_INFERENCE_THREADS=1`, BLAS threads 1, shared rate limit 60/min). The **Live site** workflow smoke-tests `LIVE_URL` daily and on demand. Supersedes the hosting part of D5 and the Space deployment in D43; the Hugging Face deploy workflow is removed | the first deploy was refused: Hugging Face now requires PRO for Docker Spaces on free CPU, and the plan requires free hosting (rule 4). Render's free tier builds Dockerfiles from GitHub with no card; its limits (sleep after 15 min idle, ~1 min wake-up, 512 MB, a fraction of one CPU) are acceptable for a demo: the image uses ~170 MB and, limited to 0.1 CPU with one thread, an analysis takes 0.9–2.8 s. Letting Render deploy on passing checks needs no deploy token in GitHub |
| D45 | 2026-10-17 | The out-of-distribution gate is the squared Mahalanobis distance of the pooled backbone features from one Gaussian fitted on all training images (covariance shrinkage 0.01), shipped as `ood_stats.npz` (mean and precision, 6.6 MB) and computed in NumPy from the same forward pass. The threshold is the smallest validation distance that lets 99% of validation images through. `fetch-non-skin` downloads 500 deterministic CC BY 2.0 photos from the Open Images V4 validation set to report the gate; they are not committed. A rejected out-of-distribution photo gets its own message instead of "retake it". The energy gate stays supported but unused | on 497 photos against 2,527 test lesions, the energy score reached AUROC 0.919 and maximum softmax 0.875; at 99% lesion acceptance energy rejected only 18% of the photos. Mahalanobis reached AUROC 0.9998: 98.9% of test lesions pass and 99.6% of photos are rejected (the two that pass are a round car gauge, which resembles the dark circular dermoscopy border, and an icy road texture). One Gaussian matched class-conditional Gaussians on a 6,000-image probe, so the simpler one ships. The threshold uses lesion images only, but the method was chosen on the same photos, so a new photo set may show slightly lower rejection. Ordinary phone photos of skin are rejected too, which is intended (rule 1): the model has never seen them |
| D46 | 2026-10-17 | `medvision-ml robustness` scores the test split under 12 conditions (blur, brightness, JPEG, noise, rotation and a phone-like mix) applied after resizing to the model input, with the shipped temperature, malignancy threshold and out-of-distribution gate. Its safety measure is `malignant_missed`: malignant images the gate accepts but the flag misses. `medvision-ml subgroups` scores the test split by sex, age band and body site from the ISIC metadata; groups under 50 images (or 20 malignant images, for recall) report counts only. `make evaluate` runs evaluate, calibrate, fetch-non-skin, fit-ood, robustness and subgroups. Answers §13 question 3: ISIC 2019 has no skin-type label or proxy, so performance by skin type is unknown and the model card says so | a rejected image is a safe outcome (the user is told it cannot be assessed), so silent misses are what a corruption costs. Corrupting the resized input keeps the run to one pass over the test images and isolates the model from decoder differences. On model-v2.1.0: blur at sigma 2 px is rejected by the gate (97%), but Gaussian noise is not (95% accepted) and silently misses 30% (std 8) to 49% (std 20) of malignant images, against 9.5% on clean images. Malignancy recall at the shipped threshold is 0.84 for women against 0.95 for men, and 0.76 for ages 30-49. These gaps are reported, not hidden, and are inputs to Phase 9 |
| D47 | 2026-10-17 | Reject grainy images: the quality checks add a sensor-noise estimate of the 224 px model input (Immerkaer's mask, mean absolute response per 8x8 block, 5th percentile across blocks) and reject above `thresholds.max_noise` with reason `too_noisy`. `fit-ood` fits the threshold so 99% of validation images pass and `export --ood` ships it; robustness counts an image as accepted only if both gates pass. Ships in `model-v2.2.0` (same weights) | D46 found Gaussian noise was the main silent failure: the out-of-distribution gate accepted 95-99% of noisy images and 30-49% of malignant ones were missed without warning. A whole-image estimate could not separate noise from hair and texture (std-8 noise: 8.5% rejected at 99% clean acceptance); the smoothest-block percentile rejects 100% of std-8 and std-20 noise, and 54% of std-4. With both gates, silent misses under noise drop to 0% while 98.4% of clean test lesions still pass (98.9% with the out-of-distribution gate alone). TensorFlow and Pillow decoding change the estimate by 0.03 on average, so fitting in `ml/` matches serving; a parity test keeps the two implementations equal. Heavy JPEG compression is not caught (25% silent misses at quality 10) and stays a documented limitation |

---

## 13. Open questions (resolve before the phase that needs them)

1. ~~**Dataset provenance and license**~~ **Resolved (D21):** retrain on ISIC
   2019 (CC BY-NC 4.0); weights published under CC BY-NC 4.0 with attribution.
2. **Attribution and license** (Phase 1): the prototype was built with other
   contributors. Confirm the license and credit them in the README.
3. ~~**Does a skin-tone label or proxy exist**~~ **Resolved (D46):** no; ISIC
   2019 records sex, age and body site only. The gap is stated in the model card.
4. **Leakage on the real dataset** (Phase 2): superseded by D21/D22. The
   ISIC 2019 split is lesion-grouped and checked by `check-leakage`.
5. **Which baseline model to ship first** (Phase 3): the 76.3% baseline or the
   sqrt-weighted 73.9% model. Decide with the §8 selection metric after
   calibration.
6. **DVC remote** (Phase 9): choose only if the dataset can be shared.
