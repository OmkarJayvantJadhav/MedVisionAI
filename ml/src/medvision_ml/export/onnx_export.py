"""Export a Keras classifier to the ONNX artifacts the backend serves.

The export directory contains:

* ``model.onnx``: input ``image`` (N, H, W, 3) float32 raw 0-255 pixels;
  outputs ``probabilities`` (N, C) and ``feature_maps`` (N, h, w, K), the
  backbone output used for Grad-CAM.
* ``head_weights.npz``: the Dense head ``kernel`` (K, C) and ``bias`` (C,). The
  backend derives logits as ``mean(feature_maps) @ kernel + bias`` and Grad-CAM
  as ``ReLU(feature_maps @ kernel[:, c])``.
* ``model_meta.json``: version, classes, input spec, calibration, thresholds and
  SHA-256 checksums of the other two files.

"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import keras
import numpy as np
import tensorflow as tf

from medvision_ml.explainability.gradcam import _head_dense, _pre_backbone_layers, head_weights
from medvision_ml.models.builder import get_backbone

LOGGER = logging.getLogger(__name__)

META_FORMAT_VERSION = 1
INPUT_NAME = "image"
PROBABILITIES = "probabilities"
FEATURE_MAPS = "feature_maps"
OPSET = 17


@dataclass(frozen=True)
class Thresholds:
    """Decision thresholds shipped with a model.

    The defaults are provisional until they are fitted on validation data
    (``fitted`` stays false until then, and the backend reports it).
    """

    low_confidence: float = 0.50
    malignant: float = 0.30
    ood_energy: float | None = None
    min_brightness: float = 20.0
    max_brightness: float = 235.0
    min_sharpness: float = 5.0
    fitted: bool = False


@dataclass(frozen=True)
class ExportResult:
    """Paths written by :func:`export_model` and the measured parity."""

    directory: Path
    meta: dict[str, Any]
    max_probability_diff: float
    files: dict[str, Path] = field(default_factory=dict)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_serving_function(model: keras.Model, image_size: int) -> Any:
    """Return a ``tf.function`` mapping raw images to probabilities and feature maps.

    Probabilities are recomputed from the pooled feature maps and the Dense
    head so they are exactly consistent with the CAM the backend derives.
    """
    backbone = get_backbone(model)
    pre_layers = _pre_backbone_layers(model, backbone)
    dense = _head_dense(model)
    kernel = tf.constant(np.asarray(dense.kernel, dtype=np.float32))
    bias = tf.constant(np.asarray(dense.bias, dtype=np.float32))

    @tf.function(  # type: ignore[untyped-decorator]
        input_signature=[tf.TensorSpec([None, image_size, image_size, 3], tf.float32)]
    )
    def serve(images: Any) -> dict[str, Any]:
        x = images
        for layer in pre_layers:
            x = layer(x, training=False)
        features = tf.cast(backbone(x, training=False), tf.float32)
        logits = tf.reduce_mean(features, axis=(1, 2)) @ kernel + bias
        return {FEATURE_MAPS: features, PROBABILITIES: tf.nn.softmax(logits)}

    return serve


def _rename_io(model_proto: Any) -> None:
    """Give the graph stable input/output names, identifying outputs by rank."""
    graph = model_proto.graph
    renames: dict[str, str] = {graph.input[0].name: INPUT_NAME}
    for output in graph.output:
        rank = len(output.type.tensor_type.shape.dim)
        renames[output.name] = {2: PROBABILITIES, 4: FEATURE_MAPS}[rank]
    if sorted(renames.values()) != sorted([INPUT_NAME, PROBABILITIES, FEATURE_MAPS]):
        raise RuntimeError(f"Unexpected ONNX graph signature: {renames}")
    for value in [*graph.input, *graph.output]:
        value.name = renames.get(value.name, value.name)
    for node in graph.node:
        node.input[:] = [renames.get(n, n) for n in node.input]
        node.output[:] = [renames.get(n, n) for n in node.output]


def convert_to_onnx(model: keras.Model, path: Path, image_size: int) -> None:
    """Convert ``model`` to an ONNX file with a dynamic batch dimension."""
    import onnx
    import tf2onnx
    from tensorflow.python.framework.convert_to_constants import (
        convert_variables_to_constants_v2,
    )

    # Freezing first turns layer variables (e.g. EfficientNet's normalisation
    # statistics) into constants; otherwise tf2onnx exposes them as graph inputs.
    frozen = convert_variables_to_constants_v2(
        build_serving_function(model, image_size).get_concrete_function()
    )
    model_proto, _ = tf2onnx.convert.from_graph_def(
        frozen.graph.as_graph_def(),
        input_names=[t.name for t in frozen.inputs],
        output_names=[t.name for t in frozen.outputs],
        opset=OPSET,
    )
    _rename_io(model_proto)
    onnx.checker.check_model(model_proto)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model_proto, path)


def onnx_probabilities(path: Path, images: np.ndarray) -> np.ndarray:
    """Run the exported model with ONNX Runtime and return its probabilities."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (probabilities,) = session.run([PROBABILITIES], {INPUT_NAME: images.astype(np.float32)})
    return np.asarray(probabilities)


def export_model(
    model: keras.Model,
    output_dir: Path,
    *,
    class_names: Sequence[str],
    malignant_classes: Sequence[str],
    model_version: str,
    backbone: str,
    image_size: int,
    thresholds: Thresholds | None = None,
    temperature: float = 1.0,
    calibration_fitted: bool = False,
    parity_tolerance: float = 1e-4,
    seed: int = 0,
) -> ExportResult:
    """Write ``model.onnx``, ``head_weights.npz`` and ``model_meta.json``.

    The export is checked against Keras on random images before the metadata
    is written.

    Raises:
        ValueError: If inputs are inconsistent with the model.
        RuntimeError: If ONNX and Keras probabilities differ by more than
            ``parity_tolerance``.
    """
    if len(class_names) != int(model.output.shape[-1]):
        raise ValueError("class_names must have one entry per model output.")
    missing = sorted(set(malignant_classes) - set(class_names))
    if missing:
        raise ValueError(f"Malignant classes not among class names: {missing}")
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"
    weights_path = output_dir / "head_weights.npz"
    meta_path = output_dir / "model_meta.json"

    convert_to_onnx(model, onnx_path, image_size)
    kernel, bias = head_weights(model)
    np.savez(weights_path, kernel=kernel, bias=bias)

    images = np.random.default_rng(seed).uniform(0, 255, (4, image_size, image_size, 3))
    expected = np.asarray(model(images.astype(np.float32), training=False), dtype=np.float64)
    max_diff = float(np.abs(onnx_probabilities(onnx_path, images) - expected).max())
    if max_diff > parity_tolerance:
        raise RuntimeError(
            f"ONNX/Keras probability mismatch {max_diff:.2e} > {parity_tolerance:.0e}"
        )

    meta: dict[str, Any] = {
        "format_version": META_FORMAT_VERSION,
        "model_version": model_version,
        "backbone": backbone,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "class_names": list(class_names),
        "malignant_classes": sorted(malignant_classes, key=list(class_names).index),
        "input": {
            "name": INPUT_NAME,
            "height": image_size,
            "width": image_size,
            "channels": 3,
            "layout": "NHWC",
            "dtype": "float32",
            "value_range": [0, 255],
            "resize": "bilinear",
        },
        "outputs": {"probabilities": PROBABILITIES, "feature_maps": FEATURE_MAPS},
        "calibration": {"temperature": temperature, "fitted": calibration_fitted},
        "thresholds": asdict(thresholds or Thresholds()),
        "parity": {"max_probability_diff": max_diff, "tolerance": parity_tolerance},
        "files": {
            onnx_path.name: {"sha256": sha256_file(onnx_path)},
            weights_path.name: {"sha256": sha256_file(weights_path)},
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    LOGGER.info(
        "Exported %s to %s (max probability diff %.2e)", model_version, output_dir, max_diff
    )
    return ExportResult(
        directory=output_dir,
        meta=meta,
        max_probability_diff=max_diff,
        files={"onnx": onnx_path, "head_weights": weights_path, "meta": meta_path},
    )
