"""Saving and loading Keras models."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import keras

LOGGER = logging.getLogger(__name__)


def save_model(model: keras.Model, path: Path, class_names: list[str]) -> Path:
    """Save a model in ``.keras`` format with its class names beside it.

    Returns:
        Path of the written class-name file (``<stem>_class_names.json``).
    """
    if path.suffix != ".keras":
        raise ValueError("Models must be saved with the .keras suffix.")
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    names_path = path.with_name(f"{path.stem}_class_names.json")
    names_path.write_text(json.dumps(class_names, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Saved model to %s", path)
    return names_path


_REDUCED_PRECISION = frozenset({"mixed_float16", "mixed_bfloat16", "float16", "bfloat16"})


def _float32_config(value: Any) -> Any:
    """Return a copy of a Keras config with every reduced-precision dtype policy set to float32."""
    if isinstance(value, dict):
        return {k: _float32_config(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return type(value)(_float32_config(v) for v in value)
    if isinstance(value, str) and value in _REDUCED_PRECISION:
        return "float32"
    return value


def to_float32(model: keras.Model) -> keras.Model:
    """Rebuild a model trained with mixed precision so it computes in float32.

    Mixed precision keeps the weights in float32 but computes in float16, and a
    saved model keeps that policy. Inference, evaluation and export must match
    the float32 ONNX model that is served, so the architecture is rebuilt with
    float32 layers and the (already float32) weights are loaded into it unchanged.
    Models without reduced-precision layers are returned as they are.
    """
    config = model.get_config()
    converted = _float32_config(config)
    if converted == config:
        return model
    rebuilt = model.__class__.from_config(converted)
    # Copy the weights through Keras's own loader. `set_weights` assigns the
    # variables but skips the load hooks, so a Normalization layer kept its
    # initial statistics and silently stopped normalising the input (D40).
    with tempfile.TemporaryDirectory() as tmp:
        weights = Path(tmp) / "model.weights.h5"
        model.save_weights(weights)
        rebuilt.load_weights(weights)
    return rebuilt


def load_model(path: Path, *, trust_legacy_lambda: bool = False) -> keras.Model:
    """Load a ``.keras`` model for inference, computing in float32.

    Args:
        path: Model file.
        trust_legacy_lambda: Allow models that contain a ``Lambda`` preprocessing
            layer (the prototype's format). Deserialising a ``Lambda`` layer can
            execute code, so only enable this for files you produced yourself.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    kwargs: dict[str, Any] = {"compile": False}
    if trust_legacy_lambda:
        kwargs["safe_mode"] = False
        kwargs["custom_objects"] = {
            "preprocess_input": keras.applications.efficientnet.preprocess_input
        }
    return to_float32(keras.models.load_model(path, **kwargs))


def load_class_names(path: Path) -> list[str]:
    """Read a JSON list of class names.

    Raises:
        ValueError: If the file does not contain at least two unique names.
    """
    names = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(names, list)
        or len(names) < 2
        or not all(isinstance(n, str) and n for n in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError(f"{path} must contain a list of at least two unique class names.")
    return names
