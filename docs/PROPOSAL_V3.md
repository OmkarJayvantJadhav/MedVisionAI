# MedVision AI v3: skin triage from phone and dermoscopy photos

Draft proposal for the next major version. Once agreed, it replaces §1 of
[PROJECT_PLAN.md](PROJECT_PLAN.md) and is recorded in the decision log.

---

## 1. Problem

Most people who notice a new spot, rash or patch on their skin have only a
phone camera and one question: **"how soon should a doctor see this?"**
Dermatologists are scarce, especially outside cities, so the answer decides
whether a skin cancer or a spreading infection is seen in days or in months.

Tools that answer from a photo exist, but they are hard to trust:

- **They answer every photo**, including blurry ones, non-skin images and
  conditions they were never trained on.
- **They name a single disease** with a confident percentage, when even
  dermatologists often disagree on a photo alone.
- **They are rarely measured on darker skin**, where redness, inflammation and
  pigment look different and published models are known to perform worse.
- **They hide what they missed**: accuracy is reported, the number of urgent
  cases sent home is not.
- **They do not show what they looked at.**

MedVision v2 solved these problems for **dermoscopy images of a single
lesion**. But dermoscopy needs a dermatoscope, which most people do not own,
and it covers only 8 lesion types, not the rashes and infections people
photograph most often.

## 2. Objective

Build a **skin triage tool** that takes a phone photo or a dermoscopy image of
a skin problem and:

1. says **how urgently it should be seen by a doctor**;
2. lists the **conditions it most resembles**, as a set sized to contain the
   right answer 90% of the time, never as a diagnosis;
3. shows **where it looked**;
4. **refuses** photos it cannot judge and says why;
5. reports its performance **by skin tone**, including on external data from
   people with darker skin.

It is not a medical device, makes no diagnosis, and says so on every screen.

### Success criteria (ranked; a lower item never trades away a higher one)

| # | Objective | Criterion (fitted on validation, reported on test) |
|---|---|---|
| 1 | Catch urgent cases | Recall of urgent cases (possible skin cancer, urgent infections and rashes) ≥ 0.90 for each input type |
| 2 | Fair across skin tones | Urgent-case recall on Fitzpatrick V–VI within 0.05 of Fitzpatrick I–II, checked on SCIN, DDI and PASSION |
| 3 | Honest confidence | Calibration error (ECE) ≤ 0.05 for each input type |
| 4 | Refuse what it cannot judge | ≥ 99% of non-skin photos rejected while ≥ 97% of valid skin photos pass |
| 5 | Honest differential | The list of likely conditions contains the reference label ≥ 90% of the time (conformal coverage) |
| 6 | Explain itself | A heatmap for every answer, exact for the shipped head (parity test) |
| 7 | Run anywhere | CPU-only inference ≤ 200 ms per image in one container under 512 MB, no external calls |
| 8 | Classify well | Macro F1 and top-3 accuracy per condition group reported; targets set after the baseline |

## 3. Scope

### In scope

- **Input:** one photo of one skin problem, taken with a phone (close-up) or
  through a dermatoscope. JPEG, PNG or WebP, ≤ 8 MB.
- **Adults** (18 and over), because the training data is mostly adult.
- **Conditions**, grouped under one taxonomy:

| Group | Conditions | Main data |
|---|---|---|
| Skin cancer and pre-cancer | Melanoma, basal cell carcinoma, squamous cell carcinoma, actinic keratosis | ISIC 2019, PAD-UFES-20, MIDAS |
| Benign lesions | Melanocytic nevus, benign/seborrheic keratosis, dermatofibroma, vascular lesion | ISIC 2019, PAD-UFES-20, MIDAS |
| Inflammatory rashes | Eczema, allergic and irritant contact dermatitis, urticaria (hives), psoriasis, drug rash | SCIN, Fitzpatrick17k |
| Infections | Tinea (fungal), impetigo, folliculitis, herpes zoster (shingles), herpes simplex | SCIN, Fitzpatrick17k |
| Other common | Insect bite, acne | SCIN, Fitzpatrick17k |
| Other / not sure | Everything else | All |

- **Urgency levels:** see a doctor soon · book an appointment · likely minor,
  watch it · not assessed. Each condition maps to one level; the mapping is
  reviewed by a clinician before release.
- **A short symptom checklist** (bleeding, rapid change, fever, spreading,
  pain) that can only **raise** the urgency, using fixed rules.

### Out of scope

| Excluded | Reason |
|---|---|
| Diagnosis, treatment or medical advice | Only a clinician can diagnose; the tool supports the decision to seek care |
| Children | Little paediatric training data; PASSION is used only to measure, not to claim support |
| Nails, scalp, mouth, eyes and genital areas | Very different appearance and too little data; genital photos also raise safeguarding concerns |
| Conditions outside the taxonomy | Too few labelled examples; they go to "other / not sure" with a "see a doctor" answer |
| Generated text | Summaries come from a fixed template |
| Accounts, stored images, history | No patient data is kept |
| Commercial use | Several datasets are licensed for non-commercial use only |

## 4. Data

| Dataset | Images | What it adds | Licence | Role |
|---|---|---|---|---|
| **ISIC 2019** | 25,331 | Dermoscopy, 8 lesion classes | CC BY-NC 4.0 | Train / val / test (as in v2) |
| **PAD-UFES-20** | 2,298 | Smartphone photos of 6 lesion classes, skin type recorded | CC BY 4.0 (to confirm) | Train / val / test |
| **MIDAS** (Stanford AIMI) | 3,830 | The same lesions photographed with a phone-style camera and with dermoscopy | Non-commercial research, access agreement | Train / val / test, if the agreement allows publishing a model |
| **SCIN** | 10,407 (5,033 cases, 3,061 labelled) | Phone photos of rashes and infections, symptoms, Fitzpatrick and Monk skin tone | SCIN Data Use License (attribution, no re-identification) | Train / val / test |
| **Fitzpatrick17k** | 16,577 | 114 conditions across all skin types | CC BY-NC-SA 3.0, images by request | Evaluation first; training only if share-alike terms are accepted for the weights |
| **DDI** (Stanford AIMI) | 656 | Biopsy-proven lesions, deliberately balanced light and dark skin | Non-commercial research, access agreement | **External test only** |
| **PASSION** | 4,901 | Eczema, fungal infection, scabies, impetigo on dark skin (Sub-Saharan Africa) | Reported CC BY-NC 4.0 (to confirm) | **External test only** |

Rules carried over from v2:

- Splits are grouped by patient, lesion or case (ISIC `lesion_id`, PAD-UFES
  patient, SCIN `case_id`), and near-duplicates never cross splits;
  `check-leakage` covers every dataset.
- External test sets are never used for training, threshold fitting or model
  selection.
- SCIN labels are weighted lists from several dermatologists; training uses
  those weights instead of forcing a single answer.
- Cases without a gradable label are excluded and counted.
- Every dataset is credited in `docs/DATASETS.md`; published weights carry the
  most restrictive licence among their training data.

## 5. Architecture (changes from v2)

```
photo ─► quality checks (brightness, blur, noise, compression)
      ─► image-type check: dermoscopy │ phone close-up │ not skin (rejected)
      ─► one backbone ─► pooled features ─► condition head (pool → dense)
                                         └► out-of-distribution distance (per image type)
      ─► calibration (temperature per image type)
      ─► conformal set of likely conditions
      ─► urgency rules (condition set + symptom checklist; can only go up)
      ─► answer: urgency · likely conditions · heatmap · reasons · disclaimer
```

- **One backbone for both image types** keeps the container small. The head
  stays `pool → dense`, so the heatmap stays exact.
- **Thresholds, calibration and the out-of-distribution gate are fitted
  separately for each image type**, because phone photos and dermoscopy differ.
- **The urgency answer is decided by the most urgent condition in the
  conformal set**, so an uncertain case is escalated, never reassured.
- Serving stays ONNX Runtime and NumPy, with no TensorFlow and no external
  calls.

## 6. Phases

| Phase | Outcome |
|---|---|
| 12 | Data access and audit: request DDI and MIDAS, Fitzpatrick17k images and PASSION; confirm every licence; count images per condition, skin tone and image type |
| 13 | Taxonomy and urgency mapping across all datasets, reviewed by a clinician |
| 14 | Phone-photo lesion model (PAD-UFES-20 + MIDAS) with its own gates; baseline of the v2 model on phone photos |
| 15 | Common-condition model (SCIN + Fitzpatrick17k) merged into one taxonomy |
| 16 | Conformal sets, urgency rules and the symptom checklist |
| 17 | Fairness and external tests on DDI and PASSION; model card v3 |
| 18 | Interface (photo guidance, urgency first, likely conditions, symptom checklist) and release `model-v3.0.0` |

## 7. Risks

| Risk | Mitigation |
|---|---|
| Licences do not allow publishing weights trained on a dataset | Use that dataset for evaluation only |
| Few examples for many conditions (only 13 SCIN conditions have ≥ 50 cases) | A small taxonomy plus "other / not sure"; report per-condition counts |
| Label noise (dermatologists disagree on photos) | Train on weighted labels; report agreement as the ceiling |
| Worse results on darker skin | Measure on SCIN V–VI, DDI and PASSION; publish the gap; never claim parity that is not measured |
| No clinician in the team | The urgency mapping is an open question until reviewed |
| Regulation (software suggesting conditions can be a medical device) | No diagnosis, urgency-first wording, disclaimer on every screen, non-commercial |

## 8. Open questions

1. Who reviews the taxonomy and urgency mapping clinically?
2. Do the Stanford AIMI agreements (DDI, MIDAS) allow publishing a model
   trained on MIDAS, or only research use?
3. Are Fitzpatrick17k weights acceptable under share-alike, or is it
   evaluation only?
4. Is one backbone enough for both image types, or do dermoscopy and phone
   photos need separate models?
