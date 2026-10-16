"""Fixtures: a tiny ONNX model with the real model's structure, built in memory.

``feature_maps`` = 8x8 average pooling of the 32x32 input (so channel k of the
feature maps is the local mean of colour channel k), and the head is
``mean -> Dense -> softmax``. With the kernel below, red images classify as
Acne, green as Melanoma and blue as Nevus.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

CLASSES = ["Acne", "Melanoma", "Nevus", "Seborrheic Keratosis"]
SIZE = 32
KERNEL = np.array(
    [[0.08, -0.04, -0.04, 0.0], [-0.04, 0.08, -0.04, 0.0], [-0.04, -0.04, 0.08, 0.0]],
    dtype=np.float32,
)
BIAS = np.zeros(len(CLASSES), dtype=np.float32)


def build_onnx(kernel: np.ndarray = KERNEL, bias: np.ndarray = BIAS) -> bytes:
    nodes = [
        helper.make_node("Transpose", ["image"], ["nchw"], perm=[0, 3, 1, 2]),
        helper.make_node(
            "AveragePool", ["nchw"], ["pooled_nchw"], kernel_shape=[8, 8], strides=[8, 8]
        ),
        helper.make_node("Transpose", ["pooled_nchw"], ["feature_maps"], perm=[0, 2, 3, 1]),
        helper.make_node("ReduceMean", ["feature_maps"], ["gap"], axes=[1, 2], keepdims=0),
        helper.make_node("MatMul", ["gap", "kernel"], ["matmul"]),
        helper.make_node("Add", ["matmul", "bias"], ["logits"]),
        helper.make_node("Softmax", ["logits"], ["probabilities"], axis=-1),
    ]
    graph = helper.make_graph(
        nodes,
        "fake_classifier",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [None, SIZE, SIZE, 3])],
        [
            helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [None, len(bias)]),
            helper.make_tensor_value_info("feature_maps", TensorProto.FLOAT, [None, 4, 4, 3]),
        ],
        initializer=[
            numpy_helper.from_array(kernel, "kernel"),
            numpy_helper.from_array(bias, "bias"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return bytes(model.SerializeToString())


def make_meta(onnx_bytes: bytes, weights_bytes: bytes, **overrides: Any) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "format_version": 1,
        "model_version": "test-v0.0.1",
        "backbone": "fake",
        "class_names": CLASSES,
        "malignant_classes": ["Melanoma"],
        "input": {
            "name": "image",
            "height": SIZE,
            "width": SIZE,
            "channels": 3,
            "layout": "NHWC",
            "dtype": "float32",
            "value_range": [0, 255],
            "resize": "bilinear",
        },
        "outputs": {"probabilities": "probabilities", "feature_maps": "feature_maps"},
        "calibration": {"temperature": 1.0, "fitted": False},
        "thresholds": {
            "low_confidence": 0.5,
            "malignant": 0.3,
            "ood_energy": None,
            "min_brightness": 20.0,
            "max_brightness": 235.0,
            "min_sharpness": 5.0,
            "fitted": False,
        },
        "files": {
            "model.onnx": {"sha256": hashlib.sha256(onnx_bytes).hexdigest()},
            "head_weights.npz": {"sha256": hashlib.sha256(weights_bytes).hexdigest()},
        },
    }
    meta.update(overrides)
    return meta


def write_model_dir(directory: Path, **meta_overrides: Any) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    onnx_bytes = build_onnx()
    buffer = io.BytesIO()
    np.savez(buffer, kernel=KERNEL, bias=BIAS)
    weights_bytes = buffer.getvalue()
    (directory / "model.onnx").write_bytes(onnx_bytes)
    (directory / "head_weights.npz").write_bytes(weights_bytes)
    meta = make_meta(onnx_bytes, weights_bytes, **meta_overrides)
    (directory / "model_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return directory


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    return write_model_dir(tmp_path / "model")


ImageFactory = Callable[..., bytes]


def textured(
    colour: tuple[int, int, int], size: tuple[int, int] = (64, 64), seed: int = 0
) -> np.ndarray:
    """A colour plus mild noise, so sharpness checks pass."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(-25, 26, size=(size[1], size[0], 3))
    return np.clip(np.array(colour) + noise, 0, 255).astype(np.uint8)


def encode(array: np.ndarray, fmt: str = "PNG", **kwargs: Any) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


@pytest.fixture
def image_bytes() -> ImageFactory:
    def _make(
        colour: tuple[int, int, int] = (200, 60, 60),
        size: tuple[int, int] = (64, 64),
        fmt: str = "PNG",
        seed: int = 0,
    ) -> bytes:
        return encode(textured(colour, size, seed), fmt)

    return _make
