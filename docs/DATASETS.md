# Datasets

## Training data: ISIC 2019

| | |
|---|---|
| Source | ISIC 2019 Challenge, training release ([challenge.isic-archive.com/data](https://challenge.isic-archive.com/data/)) |
| Content | 25,331 dermoscopic images, 8 diagnostic classes, expert-confirmed labels |
| Components | HAM10000, BCN_20000 and MSK collections |
| Licence | **CC BY-NC 4.0**: attribution required, non-commercial use only |
| Metadata used | `lesion_id` (several photos of one lesion), age, sex, anatomical site |

Class distribution (training release):

| Class | ISIC code | Images | Malignant / pre-malignant |
|---|---|---:|:---:|
| Melanocytic Nevus | NV | 12,875 | |
| Melanoma | MEL | 4,522 | ✓ |
| Basal Cell Carcinoma | BCC | 3,323 | ✓ |
| Benign Keratosis | BKL | 2,624 | |
| Actinic Keratosis | AK | 867 | ✓ |
| Squamous Cell Carcinoma | SCC | 628 | ✓ |
| Vascular Lesion | VASC | 253 | |
| Dermatofibroma | DF | 239 | |

**Lesion leakage.** 16,459 of the 25,331 images belong to 5,059 lesions that
were photographed more than once. Splitting image by image would put photos of
the same lesion in both training and test sets and overstate accuracy, so the
split is grouped by `lesion_id` (and by near-duplicate pixels as a second line
of defence).

**Near-duplicate fingerprint.** Many ISIC 2019 images share a dark circular
border. A small hash of the whole frame mostly encodes that border and made
thousands of unrelated lesions look like copies, so the fingerprint is a
256-bit difference hash of the central 60% of the image, and two images count
as copies within 16 bits. On ISIC 2019 every pair up to 18 bits is the same
lesion photo and unrelated lesions start at 19. `split` stops with an error if
a near-duplicate group grows beyond 20 images.

**Duplicates with conflicting labels.** The source collections overlap, and a
few identical images appear under two different diagnoses. `split` keeps one
copy of each identical image, drops every copy whose label disagrees (the true
diagnosis is unknown), and records the removed files in
`outputs/split_removed.json`. Near-duplicates with different labels are kept
but always placed in the same split.

### Why ISIC 2019 and not the prototype's dataset

The prototype's dataset had no documented source or licence and mixed acne
photographs with dermoscopic images. Acne was dropped because:

- no freely licensed acne dataset with documented provenance, consent and
  dermatologist labels was found;
- acne images are ordinary camera photos while the other classes are
  dermoscopic, so a model can learn the imaging device instead of the disease.

ISIC 2019 adds Vascular Lesion and replaces Seborrheic Keratosis with the
broader Benign Keratosis class.

### Preparing the data

```bash
cd ml
curl -O https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Input.zip   # ~9.8 GB
curl -O https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_GroundTruth.csv
curl -O https://isic-challenge-data.s3.amazonaws.com/2019/ISIC_2019_Training_Metadata.csv
unzip -q ISIC_2019_Training_Input.zip

uv run medvision-ml prepare-isic2019 \
  --images ISIC_2019_Training_Input \
  --ground-truth ISIC_2019_Training_GroundTruth.csv \
  --metadata ISIC_2019_Training_Metadata.csv \
  --dest data/isic2019_raw
uv run medvision-ml split --source data/isic2019_raw     # writes data/skin/{Train,Validation,Test}
uv run medvision-ml check-leakage                        # must exit 0
```

### Training, evaluation and export

```bash
cd ml
uv run medvision-ml train --run-name isic2019-effb0
uv run medvision-ml evaluate --model artifacts/isic2019-effb0/model.keras
uv run medvision-ml calibrate --model artifacts/isic2019-effb0/model.keras
uv run medvision-ml fetch-non-skin --dest data/non_skin --count 500
uv run medvision-ml fit-ood --model artifacts/isic2019-effb0/model.keras \
  --non-skin data/non_skin \
  --calibration outputs/calibration/isic2019-effb0/calibration.json
uv run medvision-ml export --model artifacts/isic2019-effb0/model.keras \
  --version skin-effb0-v2.1.0 --output export/skin-effb0-v2.1.0 \
  --calibration outputs/calibration/isic2019-effb0/calibration.json \
  --ood outputs/ood/isic2019-effb0/ood.json
```

`calibrate` fits the temperature and the decision thresholds on the validation
split and reports them on the test split. `fit-ood` fits the
out-of-distribution gate (below) and the noise check, whose threshold lets the
same 99% of validation images through. `export --calibration --ood` ships both in
the serving files.

`make evaluate MODEL=ml/artifacts/isic2019-effb0/model.keras METADATA=<path to
ISIC_2019_Training_Metadata.csv>` regenerates every report the model card
quotes: test metrics, calibration, the out-of-distribution gate, robustness to
twelve image conditions (`medvision-ml robustness`) and results by sex, age band
and body site (`medvision-ml subgroups`). Reports are written to
`ml/outputs/<report>/<run>/`.

A GPU is used automatically when TensorFlow finds one (with mixed precision);
otherwise training runs on the CPU.

**On a free Kaggle GPU.** Import
[`notebooks/train_kaggle.ipynb`](../notebooks/train_kaggle.ipynb) into Kaggle
(*Create → New Notebook → File → Import Notebook*, then paste its GitHub URL).
In *Session options*, set *Accelerator* to **GPU T4 x2** and *Internet* to
**On** (the account must be phone-verified). Then choose *Save Version → Save &
Run All (Commit)*: the run continues with the browser closed and its results
appear in the version's *Output* tab. Every step, including the leakage check,
stops the run if it fails. Decoded images can be cached on disk with
`MEDVISION__DATA__CACHE_DIR=/path/to/cache`, because the full set needs about
12 GB of memory.

### Ordinary photos for the out-of-distribution check

The model only knows dermoscopy images, so the app rejects uploads that look
nothing like them. `fit-ood` fits a Gaussian to the pooled backbone features of
the training images and sets the rejection threshold on the squared
Mahalanobis distance so that 99% of validation images pass
(`evaluation.target_in_distribution_acceptance`). The threshold uses lesion
images only.

To report how well the gate works, `fetch-non-skin` downloads 500 everyday
photos from the [Open Images V4](https://storage.googleapis.com/openimages/web/index.html)
validation set. Every image there is listed under
[CC BY 2.0](https://creativecommons.org/licenses/by/2.0/). The selection is
deterministic (the first image IDs by SHA-256), and the command writes an
`ATTRIBUTION.csv` with each photo's author, title and source page. The photos
are used for evaluation only and are not committed or redistributed. Some of
them show people, so they also check that ordinary pictures of skin, as
opposed to dermoscopy close-ups, are rejected.

### Sample images in the web app

`frontend/public/samples/` holds three ISIC 2019 images from the held-out test
split (never used in training), resized to 640 px, so visitors can try the app
without their own photo: ISIC_0000140 (melanoma), ISIC_0024885 (basal cell
carcinoma) and ISIC_0000082 (melanocytic nevus). They are redistributed under
CC BY-NC 4.0 with the attribution below.

### Required attribution

Any published model, figure or report derived from this data must cite:

> ISIC 2019 Challenge training data. Tschandl P., Rosendahl C. & Kittler H.
> The HAM10000 dataset, a large collection of multi-source dermatoscopic images
> of common pigmented skin lesions. *Sci. Data* 5, 180161 (2018).
> Codella N. et al. Skin lesion analysis toward melanoma detection: a challenge
> at the 2017 ISBI, hosted by ISIC. arXiv:1710.05006 (2017).
> Combalia M. et al. BCN20000: Dermoscopic lesions in the wild.
> arXiv:1908.02288 (2019).

## Licences of derived artefacts

| Artefact | Licence | Why |
|---|---|---|
| Source code | **MIT** ([`LICENSE`](../LICENSE)) | Independent of the data |
| Trained model weights and gate statistics (`model.onnx`, `head_weights.npz`, `ood_stats.npz`) | **CC BY-NC 4.0**, with the attribution above | Derived from CC BY-NC training data |

The hosted app is non-commercial (no fees, ads or paid features), which the
licence permits. Commercial use of the weights would need a different dataset.

## Planned evaluation-only data

These are used to test generalisation and fairness, never for training, and are
never redistributed. Check each licence again before downloading.

| Dataset | Use | Notes |
|---|---|---|
| [PAD-UFES-20](https://data.mendeley.com/datasets/zr7vgbcyr2/1) | External test on smartphone (non-dermoscopic) photos: domain shift | 2,298 images, 6 overlapping classes |
| [Fitzpatrick17k](https://github.com/mattgroh/fitzpatrick17k) | Performance by skin type (plan §8, rule 6) | CC BY-NC-SA 3.0; images fetched from their original atlases |
