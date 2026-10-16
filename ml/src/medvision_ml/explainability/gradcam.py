"""Grad-CAM reference implementation (TensorFlow) and its closed-form NumPy equivalent.

Grad-CAM weights each feature map ``A_k`` of the backbone output by the mean
gradient of the class score ``y_c`` with respect to it. With a
``global-average-pool -> Dense`` head and ``y_c`` taken as the pre-softmax
logit, that gradient is ``W[k, c] / (H * W)`` at every position, so

    Grad-CAM_c = ReLU(sum_k W[k, c] * A_k) / (H * W)

which :func:`cam_from_features` computes without TensorFlow. The serving
backend uses that formula; :func:`gradcam` is the reference it is tested
against.

A heatmap shows which image regions drove the model's score. It is not
evidence of a diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import keras
import numpy as np
import tensorflow as tf
from numpy.typing import NDArray

from medvision_ml.models.builder import get_backbone

FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class GradCamResult:
    """A normalised heatmap at feature-map resolution plus the model outputs."""

    heatmap: FloatArray
    class_index: int
    probabilities: FloatArray
    feature_maps: FloatArray


def _head_dense(model: keras.Model) -> keras.layers.Dense:
    dense = [layer for layer in model.layers if isinstance(layer, keras.layers.Dense)]
    if len(dense) != 1:
        raise ValueError("Grad-CAM expects exactly one Dense layer in the classification head.")
    if not any(isinstance(layer, keras.layers.GlobalAveragePooling2D) for layer in model.layers):
        raise ValueError("Grad-CAM expects a GlobalAveragePooling2D head.")
    return dense[0]


def _pre_backbone_layers(model: keras.Model, backbone: keras.Model) -> list[keras.layers.Layer]:
    layers = [layer for layer in model.layers if not isinstance(layer, keras.layers.InputLayer)]
    return layers[: layers.index(backbone)]


def head_weights(model: keras.Model) -> tuple[FloatArray, FloatArray]:
    """Return the classification head's Dense ``(kernel, bias)`` as float32 arrays."""
    dense = _head_dense(model)
    kernel, bias = dense.get_weights()
    return np.asarray(kernel, dtype=np.float32), np.asarray(bias, dtype=np.float32)


def normalise(heatmap: NDArray[Any]) -> FloatArray:
    """Scale a non-negative heatmap to [0, 1]; an all-zero map stays all zero."""
    peak = float(np.max(heatmap))
    scaled = heatmap / peak if peak > 0 else np.zeros_like(heatmap)
    return np.asarray(scaled, dtype=np.float32)


def cam_from_features(
    feature_maps: NDArray[Any], kernel: NDArray[Any], class_index: int
) -> FloatArray:
    """Closed-form Grad-CAM from backbone feature maps ``(H, W, K)`` and Dense kernel ``(K, C)``."""
    cam = np.tensordot(
        feature_maps.astype(np.float32), kernel[:, class_index].astype(np.float32), axes=([2], [0])
    )
    return normalise(np.maximum(cam, 0.0))


def gradcam(
    model: keras.Model, image: NDArray[Any], class_index: int | None = None
) -> GradCamResult:
    """Compute Grad-CAM with automatic differentiation.

    Args:
        model: A classifier whose layers are ``... -> backbone -> pool -> [dropout] -> Dense``.
        image: One raw RGB image ``(H, W, 3)`` with values in 0-255.
        class_index: Class to explain; defaults to the top prediction.
    """
    backbone = get_backbone(model)
    dense = _head_dense(model)
    kernel = tf.cast(dense.kernel, tf.float32)
    bias = tf.cast(dense.bias, tf.float32)

    x = tf.convert_to_tensor(image[np.newaxis], dtype=tf.float32)
    for layer in _pre_backbone_layers(model, backbone):
        x = layer(x, training=False)
    with tf.GradientTape() as tape:
        features = tf.cast(backbone(x, training=False), tf.float32)
        tape.watch(features)
        logits = tf.reduce_mean(features, axis=(1, 2)) @ kernel + bias
        probabilities = tf.nn.softmax(logits)
        target = int(tf.argmax(logits[0])) if class_index is None else class_index
        score = logits[:, target]
    gradients = tape.gradient(score, features)
    if gradients is None:
        raise RuntimeError("Could not compute gradients for Grad-CAM.")
    weights = tf.reduce_mean(gradients, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(features * weights[:, tf.newaxis, tf.newaxis, :], axis=-1))
    return GradCamResult(
        heatmap=normalise(cam[0].numpy()),
        class_index=target,
        probabilities=np.asarray(probabilities[0], dtype=np.float32),
        feature_maps=np.asarray(features[0], dtype=np.float32),
    )


def overlay(image: NDArray[Any], heatmap: NDArray[Any], alpha: float = 0.4) -> NDArray[np.uint8]:
    """Upsample ``heatmap`` to the image size, colour it and blend it over ``image``."""
    import matplotlib

    height, width = image.shape[:2]
    upsampled = tf.image.resize(heatmap[..., np.newaxis], (height, width), method="bilinear")
    coloured = matplotlib.colormaps["jet"](np.clip(upsampled.numpy()[..., 0], 0, 1))[..., :3] * 255
    blended = (1 - alpha) * image.astype(np.float32) + alpha * coloured
    return np.asarray(np.clip(blended, 0, 255), dtype=np.uint8)


def load_image(path: Path, image_size: int) -> FloatArray:
    """Load an RGB image resized to ``image_size`` with raw 0-255 float values."""
    loaded = keras.utils.load_img(path, color_mode="rgb", target_size=(image_size, image_size))
    return np.asarray(keras.utils.img_to_array(loaded), dtype=np.float32)
