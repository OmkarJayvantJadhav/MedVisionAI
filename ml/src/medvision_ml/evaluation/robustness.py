"""Robustness of the served decision rules under common image corruptions.

Each corruption is applied to the test images after they are resized to the
model input, then the model is scored with the shipped temperature and
thresholds. Besides accuracy and macro F1 the report counts what matters for
safety: how many malignant images are still flagged, and how many are missed
silently (accepted by the input gates, out-of-distribution and noise, but not
flagged).
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import keras
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from medvision_ml.calibration.fitting import calibrated_probabilities
from medvision_ml.calibration.noise import noise_level
from medvision_ml.calibration.ood import InputGates, mahalanobis
from medvision_ml.config import ProjectConfig
from medvision_ml.data.dataset import load_split
from medvision_ml.evaluation.metrics import (
    classification_metrics,
    expected_calibration_error,
    malignant_indices,
)
from medvision_ml.models.builder import LOGITS_LAYER, POOL_LAYER
from medvision_ml.models.io import load_class_names, load_model

LOGGER = logging.getLogger(__name__)

ROBUSTNESS_FILE = "robustness.json"

Batch = NDArray[np.float32]
Corrupt = Callable[[Batch, np.random.Generator], Batch]


@dataclass(frozen=True)
class Corruption:
    name: str
    description: str
    apply: Corrupt


def gaussian_blur(images: Batch, sigma: float) -> Batch:
    """Blur each image with a separable Gaussian kernel (edges padded by reflection)."""
    radius = max(1, int(np.ceil(3 * sigma)))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-(offsets**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    result = images.astype(np.float32)
    for axis in (1, 2):
        pad = [(0, 0)] * result.ndim
        pad[axis] = (radius, radius)
        padded = np.pad(result, pad, mode="reflect")
        size = result.shape[axis]
        result = sum(
            weight * np.take(padded, np.arange(i, i + size), axis=axis)
            for i, weight in enumerate(kernel)
        )
    return np.asarray(result, dtype=np.float32)


def jpeg(images: Batch, quality: int) -> Batch:
    """Re-encode each image as JPEG at ``quality``."""
    out = np.empty_like(images, dtype=np.float32)
    for i, image in enumerate(images):
        buffer = io.BytesIO()
        Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)).save(
            buffer, format="JPEG", quality=quality
        )
        with Image.open(buffer) as decoded:
            out[i] = np.asarray(decoded.convert("RGB"), dtype=np.float32)
    return out


def rotate(images: Batch, degrees: float) -> Batch:
    """Rotate each image about its centre, filling the corners with black."""
    out = np.empty_like(images, dtype=np.float32)
    for i, image in enumerate(images):
        pil = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
        out[i] = np.asarray(pil.rotate(degrees, Image.Resampling.BILINEAR), dtype=np.float32)
    return out


def phone_like(images: Batch) -> Batch:
    """Lower resolution, a warm colour cast and JPEG 60, as from a phone camera."""
    size = images.shape[1]
    small = np.empty((images.shape[0], size // 2, size // 2, 3), dtype=np.float32)
    restored = np.empty_like(images, dtype=np.float32)
    for i, image in enumerate(images):
        pil = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
        small[i] = np.asarray(pil.resize((size // 2, size // 2), Image.Resampling.BILINEAR))
        up = Image.fromarray(small[i].astype(np.uint8)).resize(
            (size, size), Image.Resampling.BILINEAR
        )
        restored[i] = np.asarray(up, dtype=np.float32)
    warm = np.clip(restored * np.array([1.06, 1.0, 0.92], dtype=np.float32), 0, 255)
    return jpeg(warm, 60)


def _scale(factor: float) -> Corrupt:
    return lambda images, _: np.clip(images * factor, 0, 255).astype(np.float32)


def _noise(std: float) -> Corrupt:
    return lambda images, rng: np.clip(images + rng.normal(0, std, images.shape), 0, 255).astype(
        np.float32
    )


CORRUPTIONS: tuple[Corruption, ...] = (
    Corruption("clean", "unchanged test images", lambda images, _: images),
    Corruption("blur_1", "Gaussian blur, sigma 1 px", lambda x, _: gaussian_blur(x, 1.0)),
    Corruption("blur_2", "Gaussian blur, sigma 2 px", lambda x, _: gaussian_blur(x, 2.0)),
    Corruption("dark", "brightness x0.6", _scale(0.6)),
    Corruption("bright", "brightness x1.4", _scale(1.4)),
    Corruption("jpeg_30", "JPEG quality 30", lambda x, _: jpeg(x, 30)),
    Corruption("jpeg_10", "JPEG quality 10", lambda x, _: jpeg(x, 10)),
    Corruption("noise_8", "Gaussian noise, std 8 grey levels", _noise(8.0)),
    Corruption("noise_20", "Gaussian noise, std 20 grey levels", _noise(20.0)),
    Corruption("rotate_45", "rotated 45 degrees", lambda x, _: rotate(x, 45.0)),
    Corruption("rotate_90", "rotated 90 degrees", lambda x, _: np.rot90(x, axes=(1, 2)).copy()),
    Corruption("phone", "half resolution, warm cast, JPEG 60", lambda x, _: phone_like(x)),
)


def condition_metrics(
    logits: Any,
    labels: Any,
    class_names: list[str],
    malignant: list[str],
    *,
    temperature: float,
    malignant_threshold: float,
    accepted: Any = None,
) -> dict[str, float]:
    """Score one condition with the shipped temperature and malignancy threshold.

    ``accepted`` marks images the input gates let through (all of them when it
    is ``None``). A malignant image is *missed* when it is
    accepted but not flagged; rejected images are not missed, because the user
    is told the photo cannot be assessed.
    """
    probabilities = calibrated_probabilities(logits, temperature)
    targets = np.asarray(labels, dtype=np.int64)
    passed = np.ones(targets.size, bool) if accepted is None else np.asarray(accepted, bool)
    summary = classification_metrics(targets, probabilities, class_names)
    group = malignant_indices(class_names, malignant)
    is_malignant = np.isin(targets, group)
    flagged = probabilities[:, group].sum(axis=1) >= malignant_threshold
    missed = is_malignant & passed & ~flagged
    return {
        "accuracy": summary["accuracy"],
        "macro_f1": summary["macro_f1"],
        "ece": expected_calibration_error(targets, probabilities),
        "malignant_recall": float(flagged[is_malignant].mean()),
        "malignant_precision": float(is_malignant[flagged].mean()) if flagged.any() else 0.0,
        "malignant_missed": float(missed[is_malignant].mean()),
        "accepted_by_gates": float(passed.mean()),
    }


def robustness(
    config: ProjectConfig,
    model_path: Path,
    *,
    temperature: float,
    malignant_threshold: float,
    gates: InputGates | None = None,
    output_dir: Path | None = None,
    corruptions: tuple[Corruption, ...] = CORRUPTIONS,
    trust_legacy_lambda: bool = False,
) -> dict[str, Any]:
    """Score the test split under every corruption and write ``robustness.json``.

    Args:
        gates: The out-of-distribution and noise gates from ``read_ood``;
            without them every image counts as accepted.
    """
    class_names = load_class_names(model_path.with_name(f"{model_path.stem}_class_names.json"))
    model = load_model(model_path, trust_legacy_lambda=trust_legacy_lambda)
    scorer = keras.Model(
        model.input, [model.get_layer(POOL_LAYER).output, model.get_layer(LOGITS_LAYER).output]
    )
    gaussian = None
    if gates is not None:
        with np.load(gates.stats_path) as stats:
            gaussian = (stats["mean"].astype(np.float64), stats["precision"])
    split = load_split(config.data_dir / "Test", config.data, shuffle=False, seed=config.seed)
    if split.class_names != class_names:
        raise ValueError(f"Dataset classes {split.class_names} differ from {class_names}")

    collected: dict[str, dict[str, list[NDArray[Any]]]] = {
        c.name: {"logits": [], "accepted": []} for c in corruptions
    }
    labels: list[NDArray[np.int64]] = []
    rng = np.random.default_rng(config.seed)
    for images, batch_labels in split.dataset:
        batch = np.asarray(images, dtype=np.float32)
        labels.append(np.asarray(batch_labels, dtype=np.int64))
        for corruption in corruptions:
            corrupted = corruption.apply(batch, rng)
            pooled, logits = scorer(corrupted, training=False)
            collected[corruption.name]["logits"].append(np.asarray(logits, dtype=np.float64))
            if gates is not None and gaussian is not None:
                passed = mahalanobis(np.asarray(pooled), *gaussian) <= gates.ood_threshold
                if gates.max_noise is not None:
                    passed &= noise_level(corrupted) <= gates.max_noise
                collected[corruption.name]["accepted"].append(passed)
    targets = np.concatenate(labels)

    conditions: dict[str, Any] = {}
    for corruption in corruptions:
        parts = collected[corruption.name]
        conditions[corruption.name] = {
            "description": corruption.description,
            **condition_metrics(
                np.concatenate(parts["logits"]),
                targets,
                class_names,
                list(config.evaluation.malignant_classes),
                temperature=temperature,
                malignant_threshold=malignant_threshold,
                accepted=np.concatenate(parts["accepted"]) if gates is not None else None,
            ),
        }
    report: dict[str, Any] = {
        "model": str(model_path),
        "split": "Test",
        "n_samples": int(targets.size),
        "temperature": temperature,
        "malignant_threshold": malignant_threshold,
        "gates": {
            "ood": gates is not None,
            "noise": gates is not None and gates.max_noise is not None,
        },
        "conditions": conditions,
    }
    out = output_dir or config.output_dir / "robustness" / model_path.parent.name
    out.mkdir(parents=True, exist_ok=True)
    (out / ROBUSTNESS_FILE).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Robustness report for %d conditions: %s", len(conditions), out)
    return report
