"""Typed project configuration loaded from YAML with environment overrides.

Loading a configuration has no side effects: no directories are created and no
frameworks are imported. Call :func:`load_config` explicitly and pass the result
to the code that needs it.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Literal, TypeVar, cast, get_args, get_type_hints

import yaml

ENV_PREFIX = "MEDVISION__"
ML_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ML_ROOT / "configs" / "skin.yaml"

Backbone = Literal["efficientnetb0", "efficientnetv2s", "convnext_tiny", "mobilenetv2"]
ClassWeighting = Literal["none", "balanced", "sqrt"]


class ConfigError(ValueError):
    """Raised when a configuration file or override is invalid."""


@dataclass(frozen=True)
class PathsConfig:
    data_dir: Path = Path("data/skin")
    output_dir: Path = Path("outputs")
    model_dir: Path = Path("artifacts")


@dataclass(frozen=True)
class DataConfig:
    image_size: int = 224
    batch_size: int = 32
    cache: bool = True
    # Cache decoded images in files under this folder instead of in memory.
    # ISIC 2019 needs about 12 GB, more than many training machines' RAM.
    cache_dir: Path | None = None
    shuffle_buffer_size: int = 1024
    split_ratios: tuple[float, float, float] = (0.80, 0.10, 0.10)
    # Hamming distance out of 256 bits; copies of one lesion photo measured up
    # to 18 on ISIC 2019, unrelated lesions from 19 (D37).
    near_duplicate_max_distance: int = 16

    def __post_init__(self) -> None:
        if self.image_size < 32:
            raise ConfigError("data.image_size must be at least 32.")
        if self.batch_size < 1:
            raise ConfigError("data.batch_size must be positive.")
        if len(self.split_ratios) != 3 or any(r < 0 for r in self.split_ratios):
            raise ConfigError("data.split_ratios must be three non-negative numbers.")
        if abs(sum(self.split_ratios) - 1.0) > 1e-6:
            raise ConfigError("data.split_ratios must sum to 1.")
        if not 0 <= self.near_duplicate_max_distance <= 256:
            raise ConfigError("data.near_duplicate_max_distance must be within [0, 256].")


@dataclass(frozen=True)
class AugmentationConfig:
    horizontal_flip: bool = True
    rotation: float = 0.15
    zoom: float = 0.20
    contrast: float = 0.15


@dataclass(frozen=True)
class ModelConfig:
    backbone: Backbone = "efficientnetb0"
    dropout: float = 0.30
    pretrained: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.dropout < 1:
            raise ConfigError("model.dropout must be within [0, 1).")


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 50
    learning_rate: float = 1e-4
    fine_tune_epochs: int = 15
    fine_tune_learning_rate: float = 1e-5
    fine_tune_layers: int = 40
    class_weighting: ClassWeighting = "sqrt"
    mixed_precision: bool = True
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 4
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 1e-7

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ConfigError("training.epochs must be positive.")
        if self.fine_tune_epochs < 0:
            raise ConfigError("training.fine_tune_epochs must be non-negative.")
        if self.learning_rate <= 0 or self.fine_tune_learning_rate <= 0:
            raise ConfigError("Learning rates must be positive.")
        if not 0 < self.reduce_lr_factor < 1:
            raise ConfigError("training.reduce_lr_factor must be within (0, 1).")


@dataclass(frozen=True)
class EvaluationConfig:
    malignant_classes: tuple[str, ...] = (
        "Actinic Keratosis",
        "Basal Cell Carcinoma",
        "Melanoma",
        "Squamous Cell Carcinoma",
    )
    calibration_bins: int = 15
    # The malignant-probability threshold is fitted so that this share of
    # malignant validation images is flagged (plan §8.4).
    target_malignant_recall: float = 0.90
    # Images whose top probability falls below the fitted threshold are reported
    # as low confidence; the rest reach this accuracy on validation.
    target_confident_accuracy: float = 0.85

    def __post_init__(self) -> None:
        if self.calibration_bins < 1:
            raise ConfigError("evaluation.calibration_bins must be positive.")
        for name in ("target_malignant_recall", "target_confident_accuracy"):
            if not 0 < getattr(self, name) <= 1:
                raise ConfigError(f"evaluation.{name} must be within (0, 1].")


@dataclass(frozen=True)
class ProjectConfig:
    seed: int = 42
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    root: Path = ML_ROOT

    def resolve(self, path: Path) -> Path:
        """Resolve a configured path relative to the configuration root."""
        return path if path.is_absolute() else self.root / path

    @property
    def data_dir(self) -> Path:
        return self.resolve(self.paths.data_dir)

    @property
    def output_dir(self) -> Path:
        return self.resolve(self.paths.output_dir)

    @property
    def model_dir(self) -> Path:
        return self.resolve(self.paths.model_dir)


T = TypeVar("T")


def _coerce(value: Any, annotation: Any, key: str) -> Any:
    """Convert a raw YAML or environment value to the annotated field type."""
    origin = getattr(annotation, "__origin__", None)
    if annotation == Path | None:
        return None if value in (None, "", "null", "none") else Path(str(value))
    if annotation is Path:
        return Path(str(value))
    if annotation is bool:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        raise ConfigError(f"{key} must be a boolean, got {value!r}.")
    if annotation in (int, float):
        if isinstance(value, bool):
            raise ConfigError(f"{key} must be a number, got {value!r}.")
        try:
            number = annotation(value)
        except (TypeError, ValueError) as error:
            raise ConfigError(f"{key} must be a number, got {value!r}.") from error
        if annotation is int and isinstance(value, float) and not value.is_integer():
            raise ConfigError(f"{key} must be an integer, got {value!r}.")
        return number
    if origin is Literal:
        allowed = get_args(annotation)
        text = str(value).strip().lower()
        if text not in allowed:
            raise ConfigError(f"{key} must be one of {', '.join(allowed)}; got {value!r}.")
        return text
    if origin is tuple:
        items = yaml.safe_load(value) if isinstance(value, str) else value
        if not isinstance(items, list | tuple):
            raise ConfigError(f"{key} must be a list.")
        args = get_args(annotation)
        item_type = args[0]
        return tuple(_coerce(item, item_type, key) for item in items)
    return value


def _build(cls: type[T], raw: Mapping[str, Any], prefix: str) -> T:
    """Recursively build a frozen dataclass, rejecting unknown keys."""
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cast(Any, cls)) if f.name != "root"}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(
            f"Unknown configuration keys under '{prefix or 'root'}': {sorted(unknown)}"
        )
    kwargs: dict[str, Any] = {}
    for name in known:
        if name not in raw:
            continue
        key = f"{prefix}.{name}" if prefix else name
        annotation = hints[name]
        value = raw[name]
        if is_dataclass(annotation):
            if not isinstance(value, Mapping):
                raise ConfigError(f"{key} must be a mapping.")
            kwargs[name] = _build(cast(type[Any], annotation), value, key)
        else:
            kwargs[name] = _coerce(value, annotation, key)
    return cls(**kwargs)


def _apply_env_overrides(raw: dict[str, Any], environ: Mapping[str, str]) -> dict[str, Any]:
    """Merge ``MEDVISION__SECTION__KEY`` environment variables into raw settings."""
    merged: dict[str, Any] = {k: dict(v) if isinstance(v, Mapping) else v for k, v in raw.items()}
    for name, value in environ.items():
        if not name.startswith(ENV_PREFIX):
            continue
        parts = [p.lower() for p in name[len(ENV_PREFIX) :].split("__") if p]
        if len(parts) == 1:
            merged[parts[0]] = value
        elif len(parts) == 2:
            section = merged.setdefault(parts[0], {})
            if not isinstance(section, dict):
                raise ConfigError(f"Cannot override {name}: '{parts[0]}' is not a section.")
            section[parts[1]] = value
        else:
            raise ConfigError(f"Environment override {name} is nested too deeply.")
    return merged


def load_config(
    path: Path | str | None = None,
    environ: Mapping[str, str] | None = None,
) -> ProjectConfig:
    """Load and validate a project configuration.

    Args:
        path: YAML file to read. Defaults to ``ml/configs/skin.yaml``.
        environ: Environment mapping for overrides. Defaults to ``os.environ``.

    Returns:
        A validated, immutable :class:`ProjectConfig` whose relative paths
        resolve against the directory above ``configs/``.

    Raises:
        ConfigError: If the file is malformed or a value is invalid.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except OSError as error:
        raise ConfigError(f"Cannot read configuration file: {config_path}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"Invalid YAML in configuration file: {config_path}") from error
    if not isinstance(loaded, dict):
        raise ConfigError("The configuration file must contain a mapping at the top level.")

    raw = _apply_env_overrides(loaded, os.environ if environ is None else environ)
    config = _build(ProjectConfig, raw, "")
    config = replace(config, root=config_path.resolve().parent.parent)
    if config.data.cache_dir is not None:
        config = replace(
            config, data=replace(config.data, cache_dir=config.resolve(config.data.cache_dir))
        )
    return config
