"""Saving and loading Keras models."""

from __future__ import annotations

import json
import logging
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


def load_model(path: Path, *, trust_legacy_lambda: bool = False) -> keras.Model:
    """Load a ``.keras`` model for inference.

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
    return keras.models.load_model(path, **kwargs)


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
