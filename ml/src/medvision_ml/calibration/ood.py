"""Out-of-distribution gate: Mahalanobis distance of pooled backbone features.

A single Gaussian is fitted to the pooled backbone features of the training
images. The squared Mahalanobis distance of a new image's features measures how
unlike the training data it is. The rejection threshold is the distance below
which ``target_in_distribution_acceptance`` of the validation images fall, so it
is fitted on lesion images only; a set of ordinary photos is used to report how
many of them the gate rejects and to compare against logit-based scores.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import keras
import numpy as np
import tensorflow as tf
from numpy.typing import NDArray

from medvision_ml.calibration.fitting import calibrated_probabilities
from medvision_ml.config import ProjectConfig
from medvision_ml.data.dataset import load_split
from medvision_ml.models.builder import LOGITS_LAYER, POOL_LAYER
from medvision_ml.models.io import load_model

LOGGER = logging.getLogger(__name__)

OOD_STATS_FILE = "ood_stats.npz"
OOD_REPORT_FILE = "ood.json"

FloatArray = NDArray[np.float64]


def fit_gaussian(features: Any, shrinkage: float) -> tuple[FloatArray, FloatArray]:
    """Return the mean and precision matrix of ``features`` (n, d).

    The covariance is shrunk towards ``trace / d`` times the identity, which
    keeps it invertible when ``n`` is small relative to ``d``.

    Raises:
        ValueError: If ``features`` is not a non-empty 2-D array or
            ``shrinkage`` is outside ``[0, 1]``.
    """
    data = np.asarray(features, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] < 2:
        raise ValueError("features must be (n, d) with at least two rows.")
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must be within [0, 1].")
    mean = data.mean(axis=0)
    centred = data - mean
    covariance = centred.T @ centred / data.shape[0]
    scale = float(np.trace(covariance)) / covariance.shape[0]
    covariance = (1 - shrinkage) * covariance + shrinkage * scale * np.eye(covariance.shape[0])
    precision: FloatArray = np.linalg.pinv(covariance, hermitian=True)
    return mean, precision


def mahalanobis(features: Any, mean: Any, precision: Any) -> FloatArray:
    """Squared Mahalanobis distance of every row of ``features``; higher is more unusual."""
    centred = np.asarray(features, dtype=np.float64) - np.asarray(mean, dtype=np.float64)
    result: FloatArray = np.einsum("nd,de,ne->n", centred, np.asarray(precision), centred)
    return result


def energy(logits: Any, temperature: float = 1.0) -> FloatArray:
    """Free energy ``-T * logsumexp(logits / T)``; higher is more unusual."""
    scores = np.asarray(logits, dtype=np.float64) / temperature
    top = scores.max(axis=1, keepdims=True)
    result: FloatArray = -temperature * (top[:, 0] + np.log(np.exp(scores - top).sum(axis=1)))
    return result


def auroc(in_scores: Any, out_scores: Any) -> float:
    """Probability that an out-of-distribution image scores higher (ties count half)."""
    inside = np.asarray(in_scores, dtype=np.float64)
    outside = np.asarray(out_scores, dtype=np.float64)
    if inside.size == 0 or outside.size == 0:
        raise ValueError("Both score sets must be non-empty.")
    greater = (outside[:, None] > inside[None, :]).sum()
    ties = (outside[:, None] == inside[None, :]).sum()
    return float((greater + 0.5 * ties) / (inside.size * outside.size))


def acceptance_threshold(scores: Any, acceptance: float) -> float:
    """Return the lowest observed score at or below which ``acceptance`` of ``scores`` fall.

    An observed score is used rather than an interpolated quantile, so at
    least ``acceptance`` of ``scores`` pass, and no more than needed.
    """
    if not 0 < acceptance <= 1:
        raise ValueError("acceptance must be within (0, 1].")
    values = np.sort(np.asarray(scores, dtype=np.float64))
    if values.size == 0:
        raise ValueError("scores must be non-empty.")
    return float(values[int(np.ceil(acceptance * values.size)) - 1])


def _feature_model(model: keras.Model) -> keras.Model:
    return keras.Model(
        model.input, [model.get_layer(POOL_LAYER).output, model.get_layer(LOGITS_LAYER).output]
    )


def _collect(feature_model: keras.Model, batches: Iterable[Any]) -> tuple[FloatArray, FloatArray]:
    pooled: list[FloatArray] = []
    logits: list[FloatArray] = []
    for images in batches:
        batch_pooled, batch_logits = feature_model(images, training=False)
        pooled.append(np.asarray(batch_pooled, dtype=np.float64))
        logits.append(np.asarray(batch_logits, dtype=np.float64))
    return np.concatenate(pooled), np.concatenate(logits)


def _split_features(
    feature_model: keras.Model, split_dir: Path, config: ProjectConfig
) -> tuple[FloatArray, FloatArray]:
    split = load_split(split_dir, config.data, shuffle=False, seed=config.seed)
    return _collect(feature_model, (images for images, _ in split.dataset))


def _folder_features(
    feature_model: keras.Model, folder: Path, config: ProjectConfig
) -> tuple[FloatArray, FloatArray]:
    size = config.data.image_size
    dataset = tf.keras.utils.image_dataset_from_directory(
        folder,
        labels=None,
        image_size=(size, size),
        batch_size=config.data.batch_size,
        shuffle=False,
        interpolation="bilinear",
    )
    return _collect(feature_model, dataset)


def _scores(
    pooled: FloatArray, logits: FloatArray, gaussian: tuple[FloatArray, FloatArray], t: float
) -> dict[str, FloatArray]:
    return {
        "mahalanobis": mahalanobis(pooled, *gaussian),
        "energy": energy(logits, t),
        "max_softmax": -calibrated_probabilities(logits, t).max(axis=1),
    }


def fit_ood(
    config: ProjectConfig,
    model_path: Path,
    non_skin_dir: Path,
    output_dir: Path | None = None,
    *,
    temperature: float = 1.0,
    trust_legacy_lambda: bool = False,
) -> dict[str, Any]:
    """Fit the feature Gaussian on Train and the threshold on Validation; report on Test.

    Writes ``ood_stats.npz`` (``mean``, ``precision``; read by ``export --ood``)
    and ``ood.json`` with the threshold and how it holds up on the test split
    and on the images in ``non_skin_dir``. Energy and maximum-softmax scores
    (at ``temperature``) are reported alongside for comparison.

    Raises:
        ValueError: If ``non_skin_dir`` contains no images.
    """
    targets = config.evaluation
    model = load_model(model_path, trust_legacy_lambda=trust_legacy_lambda)
    feature_model = _feature_model(model)
    out = output_dir or config.output_dir / "ood" / model_path.parent.name
    out.mkdir(parents=True, exist_ok=True)

    train_pooled, _ = _split_features(feature_model, config.data_dir / "Train", config)
    gaussian = fit_gaussian(train_pooled, targets.ood_shrinkage)
    sets = {
        "validation": _split_features(feature_model, config.data_dir / "Validation", config),
        "test": _split_features(feature_model, config.data_dir / "Test", config),
        "non_skin": _folder_features(feature_model, non_skin_dir, config),
    }
    scores = {name: _scores(*features, gaussian, temperature) for name, features in sets.items()}
    acceptance = targets.target_in_distribution_acceptance
    threshold = acceptance_threshold(scores["validation"]["mahalanobis"], acceptance)

    report: dict[str, Any] = {
        "model": str(model_path),
        "method": "mahalanobis",
        "threshold": threshold,
        "target_in_distribution_acceptance": acceptance,
        "shrinkage": targets.ood_shrinkage,
        "n_train": int(train_pooled.shape[0]),
        "splits": {
            name: {
                "n_samples": int(scores[name]["mahalanobis"].size),
                "accepted": float(np.mean(scores[name]["mahalanobis"] <= threshold)),
            }
            for name in ("validation", "test")
        },
        "non_skin": {
            "n_samples": int(scores["non_skin"]["mahalanobis"].size),
            "rejected": float(np.mean(scores["non_skin"]["mahalanobis"] > threshold)),
        },
        # Test lesions against ordinary photos; the energy and maximum-softmax
        # scores show what a logit-based gate would have achieved instead.
        "auroc_test_vs_non_skin": {
            method: auroc(scores["test"][method], scores["non_skin"][method])
            for method in ("mahalanobis", "energy", "max_softmax")
        },
        "temperature": temperature,
    }
    np.savez(
        out / OOD_STATS_FILE,
        mean=gaussian[0].astype(np.float32),
        precision=gaussian[1].astype(np.float32),
    )
    (out / OOD_REPORT_FILE).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    LOGGER.info(
        "OOD gate: threshold %.1f, test accepted %.3f, non-skin rejected %.3f; %s",
        threshold,
        report["splits"]["test"]["accepted"],
        report["non_skin"]["rejected"],
        out,
    )
    return report


def read_ood(path: Path) -> tuple[float, Path]:
    """Read the threshold from ``ood.json`` and locate its ``ood_stats.npz``.

    Raises:
        ValueError: If the report has no positive threshold or the statistics
            file is missing.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    threshold = data.get("threshold")
    if not isinstance(threshold, int | float) or threshold <= 0:
        raise ValueError(f"{path} has no positive threshold.")
    stats = path.with_name(OOD_STATS_FILE)
    if not stats.is_file():
        raise ValueError(f"{stats} is missing.")
    return float(threshold), stats
