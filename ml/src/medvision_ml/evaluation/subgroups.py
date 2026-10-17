"""Performance by patient subgroup on the test split.

ISIC 2019 records sex, approximate age and anatomical site for most images, but
no skin type, so these are the only subgroups that can be checked. Each group is
scored with the shipped temperature and malignancy threshold. Groups too small
for a stable estimate are reported with their size and no metric.
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import keras
import numpy as np
from numpy.typing import NDArray

from medvision_ml.calibration.fitting import calibrated_probabilities
from medvision_ml.config import ProjectConfig
from medvision_ml.data.dataset import load_split
from medvision_ml.evaluation.metrics import malignant_indices
from medvision_ml.models.builder import LOGITS_LAYER
from medvision_ml.models.io import load_class_names, load_model

LOGGER = logging.getLogger(__name__)

SUBGROUPS_FILE = "subgroups.json"
# Below these sizes a rate is too noisy to compare (a 95% interval is wider
# than about +/- 14 points), so only the count is reported.
MIN_IMAGES = 50
MIN_MALIGNANT = 20
UNKNOWN = "unknown"


def age_band(value: str) -> str:
    """Map an ``age_approx`` value to a band; empty or invalid values are unknown."""
    try:
        age = float(value)
    except ValueError:
        return UNKNOWN
    if age < 30:
        return "under 30"
    if age < 50:
        return "30-49"
    if age < 70:
        return "50-69"
    return "70 and over"


ATTRIBUTES: dict[str, Callable[[dict[str, str]], str]] = {
    "sex": lambda row: row.get("sex") or UNKNOWN,
    "age": lambda row: age_band(row.get("age_approx", "")),
    "site": lambda row: row.get("anatom_site_general") or UNKNOWN,
}


def read_metadata(path: Path) -> dict[str, dict[str, str]]:
    """Read the ISIC metadata CSV, keyed by image ID."""
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["image"]: row for row in csv.DictReader(handle)}


def group_metrics(
    probabilities: Any, labels: Any, malignant: list[int], malignant_threshold: float
) -> dict[str, Any]:
    """Accuracy and malignancy-flag performance for one subgroup."""
    probs = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    is_malignant = np.isin(targets, malignant)
    flagged = probs[:, malignant].sum(axis=1) >= malignant_threshold
    result: dict[str, Any] = {
        "n_images": int(targets.size),
        "n_malignant": int(is_malignant.sum()),
        "accuracy": None,
        "malignant_recall": None,
        "malignant_precision": None,
    }
    if targets.size >= MIN_IMAGES:
        result["accuracy"] = float((probs.argmax(axis=1) == targets).mean())
        if flagged.any():
            result["malignant_precision"] = float(is_malignant[flagged].mean())
    if is_malignant.sum() >= MIN_MALIGNANT:
        result["malignant_recall"] = float(flagged[is_malignant].mean())
    return result


def subgroup_report(
    image_ids: list[str],
    probabilities: Any,
    labels: Any,
    metadata: dict[str, dict[str, str]],
    malignant: list[int],
    malignant_threshold: float,
) -> dict[str, dict[str, Any]]:
    """Score every value of every attribute; images missing from the metadata are unknown."""
    probs = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    report: dict[str, dict[str, Any]] = {}
    for attribute, value_of in ATTRIBUTES.items():
        values = np.array([value_of(metadata.get(image, {})) for image in image_ids])
        report[attribute] = {
            value: group_metrics(
                probs[values == value], targets[values == value], malignant, malignant_threshold
            )
            for value in sorted(set(values.tolist()))
        }
    return report


def subgroups(
    config: ProjectConfig,
    model_path: Path,
    metadata_path: Path,
    *,
    temperature: float,
    malignant_threshold: float,
    output_dir: Path | None = None,
    trust_legacy_lambda: bool = False,
) -> dict[str, Any]:
    """Score the test split by sex, age band and anatomical site; write ``subgroups.json``."""
    class_names = load_class_names(model_path.with_name(f"{model_path.stem}_class_names.json"))
    model = load_model(model_path, trust_legacy_lambda=trust_legacy_lambda)
    logits_model = keras.Model(model.input, model.get_layer(LOGITS_LAYER).output)
    split = load_split(config.data_dir / "Test", config.data, shuffle=False, seed=config.seed)
    if split.class_names != class_names:
        raise ValueError(f"Dataset classes {split.class_names} differ from {class_names}")
    logits: list[NDArray[np.float64]] = []
    labels: list[NDArray[np.int64]] = []
    for images, batch_labels in split.dataset:
        logits.append(np.asarray(logits_model(images, training=False), dtype=np.float64))
        labels.append(np.asarray(batch_labels, dtype=np.int64))
    image_ids = [Path(path).stem for path in split.file_paths]
    metadata = read_metadata(metadata_path)
    malignant = malignant_indices(class_names, config.evaluation.malignant_classes)

    report: dict[str, Any] = {
        "model": str(model_path),
        "split": "Test",
        "malignant_threshold": malignant_threshold,
        "min_images": MIN_IMAGES,
        "min_malignant": MIN_MALIGNANT,
        "matched_metadata": sum(image in metadata for image in image_ids),
        "n_samples": len(image_ids),
        "skin_type": "not recorded in ISIC 2019; performance by skin type is unknown",
        "groups": subgroup_report(
            image_ids,
            calibrated_probabilities(np.concatenate(logits), temperature),
            np.concatenate(labels),
            metadata,
            malignant,
            malignant_threshold,
        ),
    }
    out = output_dir or config.output_dir / "subgroups" / model_path.parent.name
    out.mkdir(parents=True, exist_ok=True)
    (out / SUBGROUPS_FILE).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Subgroup report: %s", out)
    return report
