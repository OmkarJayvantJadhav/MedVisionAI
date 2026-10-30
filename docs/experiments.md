# Phase 9 experiments: a stronger model

Goal (plan §9, objective 8): better separation of the eight lesion types, so the
referral flag keeps catching 9 in 10 malignant lesions while flagging fewer
lesions overall. Today, `model-v2.2.0` flags 49% of test images at a precision
of 0.68, with macro F1 0.51 and melanoma top-1 recall 0.43.

## How to run an experiment

1. Open `notebooks/train_kaggle.ipynb` on Kaggle (GPU T4, internet on).
2. In section 5, set `EXPERIMENT` to one of the names below.
3. **Save Version → Save & Run All (Commit).** The run continues with the
   browser closed and writes everything to the Output tab.
4. Download `reports/` from the Output tab. The numbers that decide the
   comparison are in `reports/calibration/calibration.json` (test split, at the
   fitted thresholds) and `reports/evaluation/test/metrics.json`.

The notebook clones `main`, so the training options must be merged before the
first run. Each experiment only changes configuration settings; the code path
is the one tested in CI.

## The experiment ladder

Each step adds one change to the step before, so a gain can be attributed to
it. If a step does not help, the next one builds on the best earlier step
instead.

| Name | Change | Why |
|---|---|---|
| `baseline` | The shipped recipe, retrained | Re-baseline on the current splits and code |
| `full-ft` | Fine-tune the whole backbone (not only the last 40 layers) for 25 epochs, with a cosine schedule and 2 warm-up epochs | Dermoscopy looks nothing like ImageNet; warm-up keeps full fine-tuning stable |
| `full-ft-aug` | + vertical flips, any rotation, ±10% brightness, mild saturation | A lesion has no "up"; more valid variety for rare classes. Hue is not changed, because lesion colour carries diagnostic information |
| `full-ft-aug-focal` | + focal loss (γ = 2) | Concentrates training on hard and rare examples |
| `full-ft-aug-300` | Images at 300 px instead of 224 | Fine structures (pigment network, vessels) survive resizing |
| `v2s-aug-300` | EfficientNetV2-S at 300 px, batch 16 | A larger backbone; slower on the hosted CPU |

Check the epoch time printed early in each run: a run must fit in one 12-hour
Kaggle session, and the free quota is about 30 GPU hours a week.

## How a winner is chosen

The plan's selection rule (§8.5), never accuracy alone:

1. Malignant-group recall at the fitted referral threshold stays ≥ 0.90 on test.
2. Then the highest macro F1.
3. Then the lowest calibration error (ECE).

Also record the referral rate and precision at the threshold: the point of
Phase 9 is to lower the referral rate at the same recall.

| Experiment | Malignant recall | Precision | Referral rate | Macro F1 | Melanoma recall (top-1) | ECE | Epoch time |
|---|---|---|---|---|---|---|---|
| `model-v2.2.0` (shipped) | 0.904 | 0.681 | 49% | 0.508 | 0.43 | 0.026 | |
| `baseline` | | | | | | | |
| `full-ft` | | | | | | | |
| `full-ft-aug` | | | | | | | |
| `full-ft-aug-focal` | | | | | | | |
| `full-ft-aug-300` | | | | | | | |
| `v2s-aug-300` | | | | | | | |

## Releasing a winner

The winner's notebook run already fits the temperature, both thresholds and the
input gates, and exports the release files. Before publishing:

- run `make evaluate` (robustness and subgroups) and update the model card;
- check the CPU latency locally (≤ 100 ms per image) and on Render;
- version it: same input size → `model-v2.3.0`; a different input size changes
  the input specification, so `model-v3.0.0` (plan §10);
- update `PUBLISHED_RESULTS` in `frontend/src/lib/modelResults.js` so the app
  shows the new version's results.

## Not in this round

- **Test-time augmentation** (averaging predictions over flipped and rotated
  copies). The thresholds, the calibration and the input gates are fitted on
  single predictions, so serving with averaged predictions would need the
  whole evaluation, the export and the backend changed together. It is worth
  trying once a winner is chosen.
- **Patient details** (age, sex, body site) as inputs: they change the API and
  the interface.
- **More data** (ISIC 2020): it needs cross-dataset duplicate checks before it
  can be mixed in.
