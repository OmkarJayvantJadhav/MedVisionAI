"""Loading and verifying the exported model artifacts.

Expected directory layout (written by ``medvision-ml export``)::

    model.onnx  head_weights.npz  model_meta.json  [ood_stats.npz]

``ood_stats.npz`` holds the mean and precision matrix of the pooled training
features; it is required when the metadata sets ``ood_mahalanobis``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.inference.errors import ModelArtifactError

SUPPORTED_META_FORMAT = 1
MODEL_FILE = "model.onnx"
WEIGHTS_FILE = "head_weights.npz"
META_FILE = "model_meta.json"
OOD_FILE = "ood_stats.npz"


@dataclass(frozen=True)
class Thresholds:
    low_confidence: float
    malignant: float
    min_brightness: float
    max_brightness: float
    min_sharpness: float
    fitted: bool
    ood_mahalanobis: float | None = None
    max_noise: float | None = None


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


@dataclass(frozen=True)
class ModelMeta:
    model_version: str
    backbone: str
    class_names: tuple[str, ...]
    malignant_classes: tuple[str, ...]
    image_size: int
    input_name: str
    probabilities_output: str
    feature_maps_output: str
    temperature: float
    calibration_fitted: bool
    thresholds: Thresholds
    checksums: dict[str, str]

    @property
    def malignant_indices(self) -> tuple[int, ...]:
        return tuple(self.class_names.index(name) for name in self.malignant_classes)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ModelMeta:
        """Parse and validate ``model_meta.json`` content.

        Raises:
            ModelArtifactError: If a field is missing or inconsistent.
        """
        try:
            if raw["format_version"] != SUPPORTED_META_FORMAT:
                raise ModelArtifactError(f"Unsupported meta format {raw['format_version']}.")
            spec = raw["input"]
            if spec["height"] != spec["width"] or spec["channels"] != 3:
                raise ModelArtifactError("Only square RGB inputs are supported.")
            if spec["layout"] != "NHWC" or spec["value_range"] != [0, 255]:
                raise ModelArtifactError("Expected NHWC input with raw 0-255 values.")
            class_names = tuple(raw["class_names"])
            malignant = tuple(raw["malignant_classes"])
            if len(class_names) < 2 or len(set(class_names)) != len(class_names):
                raise ModelArtifactError("class_names must hold at least two unique names.")
            if not set(malignant) <= set(class_names):
                raise ModelArtifactError("malignant_classes must be a subset of class_names.")
            temperature = float(raw["calibration"]["temperature"])
            if temperature <= 0:
                raise ModelArtifactError("Calibration temperature must be positive.")
            t = raw["thresholds"]
            thresholds = Thresholds(
                low_confidence=float(t["low_confidence"]),
                malignant=float(t["malignant"]),
                min_brightness=float(t["min_brightness"]),
                max_brightness=float(t["max_brightness"]),
                min_sharpness=float(t["min_sharpness"]),
                fitted=bool(t["fitted"]),
                ood_mahalanobis=_optional_float(t.get("ood_mahalanobis")),
                max_noise=_optional_float(t.get("max_noise")),
            )
            return cls(
                model_version=str(raw["model_version"]),
                backbone=str(raw["backbone"]),
                class_names=class_names,
                malignant_classes=malignant,
                image_size=int(spec["height"]),
                input_name=str(spec["name"]),
                probabilities_output=str(raw["outputs"]["probabilities"]),
                feature_maps_output=str(raw["outputs"]["feature_maps"]),
                temperature=temperature,
                calibration_fitted=bool(raw["calibration"]["fitted"]),
                thresholds=thresholds,
                checksums={name: str(entry["sha256"]) for name, entry in raw["files"].items()},
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ModelArtifactError(f"Invalid model metadata: {error!r}") from error


@dataclass(frozen=True)
class ModelArtifacts:
    """Verified artifacts ready to be loaded into an inference session."""

    directory: Path
    meta: ModelMeta
    onnx_path: Path
    kernel: NDArray[np.float32]
    bias: NDArray[np.float32]
    # Mean and precision of the pooled training features (out-of-distribution gate).
    ood_mean: NDArray[np.float64] | None = None
    ood_precision: NDArray[np.float64] | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_artifacts(directory: Path, *, verify_checksums: bool = True) -> ModelArtifacts:
    """Read the metadata and head weights and verify file checksums.

    Raises:
        ModelArtifactError: If files are missing, corrupted or inconsistent.
    """
    meta_path = directory / META_FILE
    try:
        meta = ModelMeta.from_dict(json.loads(meta_path.read_text(encoding="utf-8")))
    except OSError as error:
        raise ModelArtifactError(f"Cannot read {meta_path}") from error
    except json.JSONDecodeError as error:
        raise ModelArtifactError(f"{meta_path} is not valid JSON") from error

    gated = meta.thresholds.ood_mahalanobis is not None
    for name in (MODEL_FILE, WEIGHTS_FILE, OOD_FILE) if gated else (MODEL_FILE, WEIGHTS_FILE):
        path = directory / name
        if not path.is_file():
            raise ModelArtifactError(f"Missing model file: {path}")
        expected = meta.checksums.get(name)
        if expected is None:
            raise ModelArtifactError(f"No checksum recorded for {name}")
        if verify_checksums and sha256_file(path) != expected:
            raise ModelArtifactError(f"Checksum mismatch for {path}")

    with np.load(directory / WEIGHTS_FILE) as weights:
        kernel = np.asarray(weights["kernel"], dtype=np.float32)
        bias = np.asarray(weights["bias"], dtype=np.float32)
    n_classes = len(meta.class_names)
    if kernel.ndim != 2 or kernel.shape[1] != n_classes or bias.shape != (n_classes,):
        raise ModelArtifactError("Head weights do not match the number of classes.")
    if not gated:
        return ModelArtifacts(directory, meta, directory / MODEL_FILE, kernel, bias)

    with np.load(directory / OOD_FILE) as stats:
        mean = np.asarray(stats["mean"], dtype=np.float64)
        precision = np.asarray(stats["precision"], dtype=np.float64)
    features = kernel.shape[0]
    if mean.shape != (features,) or precision.shape != (features, features):
        raise ModelArtifactError("OOD statistics do not match the number of features.")
    return ModelArtifacts(
        directory,
        meta,
        directory / MODEL_FILE,
        kernel,
        bias,
        ood_mean=mean,
        ood_precision=precision,
    )
