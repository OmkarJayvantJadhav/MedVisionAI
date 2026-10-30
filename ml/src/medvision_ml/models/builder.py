"""Transfer-learning classifier construction.

Every model has the same layer layout, which the training, explainability and
export code rely on:

    image (raw RGB, 0-255)
      -> augmentation      (active only during training)
      -> [input_scaling]   (only for backbones without built-in preprocessing)
      -> backbone          (ImageNet CNN, include_top=False)
      -> pool              (global average pooling)
      -> dropout
      -> logits            (Dense, linear)
      -> probabilities     (softmax, float32)

Keeping the head as ``pool -> dropout -> logits`` means Grad-CAM for class ``c``
equals ``ReLU(sum_k W[k, c] * A_k)`` over the backbone feature maps ``A``, so the
serving path can compute explanations with NumPy alone.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import keras

from medvision_ml.config import AugmentationConfig, Backbone, ModelConfig

LOGGER = logging.getLogger(__name__)

AUGMENTATION_LAYER = "augmentation"
BACKBONE_LAYER = "backbone"
POOL_LAYER = "pool"
LOGITS_LAYER = "logits"
PROBABILITIES_LAYER = "probabilities"


@dataclass(frozen=True)
class BackboneSpec:
    """How to build one ImageNet backbone."""

    factory: Callable[..., keras.Model]
    # Scale and offset that map raw 0-255 pixels to the range the backbone
    # expects; ``None`` when the backbone includes its own preprocessing.
    scaling: tuple[float, float] | None
    extra_kwargs: dict[str, Any]


BACKBONES: dict[Backbone, BackboneSpec] = {
    "efficientnetb0": BackboneSpec(keras.applications.EfficientNetB0, None, {}),
    "efficientnetv2s": BackboneSpec(
        keras.applications.EfficientNetV2S, None, {"include_preprocessing": True}
    ),
    "convnext_tiny": BackboneSpec(
        keras.applications.ConvNeXtTiny, None, {"include_preprocessing": True}
    ),
    "mobilenetv2": BackboneSpec(keras.applications.MobileNetV2, (1 / 127.5, -1.0), {}),
}


def build_augmentation(config: AugmentationConfig) -> keras.Sequential:
    """Return the augmentation block; Keras disables it automatically at inference."""
    layers: list[keras.layers.Layer] = []
    flips = [
        m
        for m, on in (("horizontal", config.horizontal_flip), ("vertical", config.vertical_flip))
        if on
    ]
    if flips:
        layers.append(keras.layers.RandomFlip("_and_".join(flips)))
    if config.rotation > 0:
        layers.append(keras.layers.RandomRotation(config.rotation))
    if config.zoom > 0:
        layers.append(keras.layers.RandomZoom(config.zoom))
    if config.contrast > 0:
        layers.append(keras.layers.RandomContrast(config.contrast))
    if config.brightness > 0:
        layers.append(keras.layers.RandomBrightness(config.brightness, value_range=(0, 255)))
    if config.saturation > 0:
        # Keras 3.15's RandomSaturation leaves the 0-255 range (values far below
        # zero) when told the input is 0-255, but behaves on 0-1 input, so it
        # runs on a temporarily rescaled image.
        half = config.saturation / 2
        layers += [
            keras.layers.Rescaling(1 / 255, name="saturation_in"),
            keras.layers.RandomSaturation((0.5 - half, 0.5 + half), value_range=(0, 1)),
            keras.layers.Rescaling(255, name="saturation_out"),
        ]
    return keras.Sequential(layers, name=AUGMENTATION_LAYER)


def build_model(
    model_config: ModelConfig,
    augmentation_config: AugmentationConfig,
    num_classes: int,
    image_size: int,
    *,
    weights: str | None = "imagenet",
) -> keras.Model:
    """Build an uncompiled classifier with a frozen backbone.

    Args:
        model_config: Backbone choice and dropout rate.
        augmentation_config: Training-time augmentation settings.
        num_classes: Number of output classes (at least 2).
        image_size: Square input size in pixels.
        weights: ``"imagenet"`` for pretrained weights, ``None`` for random
            initialisation (used by tests).

    Raises:
        ValueError: If ``num_classes`` is below 2.
    """
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")
    spec = BACKBONES[model_config.backbone]
    input_shape = (image_size, image_size, 3)

    base = spec.factory(
        include_top=False, weights=weights, input_shape=input_shape, **spec.extra_kwargs
    )
    backbone = keras.Model(base.input, base.output, name=BACKBONE_LAYER)
    backbone.trainable = False

    inputs = keras.Input(shape=input_shape, name="image")
    x = build_augmentation(augmentation_config)(inputs)
    if spec.scaling is not None:
        scale, offset = spec.scaling
        x = keras.layers.Rescaling(scale, offset, name="input_scaling")(x)
    x = backbone(x, training=False)
    x = keras.layers.GlobalAveragePooling2D(name=POOL_LAYER)(x)
    x = keras.layers.Dropout(model_config.dropout, name="dropout")(x)
    logits = keras.layers.Dense(num_classes, name=LOGITS_LAYER)(x)
    probabilities = keras.layers.Activation("softmax", dtype="float32", name=PROBABILITIES_LAYER)(
        logits
    )
    model = keras.Model(inputs, probabilities, name=f"{model_config.backbone}_classifier")
    LOGGER.info("Built %s | input %s | classes %d", model_config.backbone, input_shape, num_classes)
    return model


def get_backbone(model: keras.Model) -> keras.Model:
    """Return the nested backbone model.

    Raises:
        ValueError: If the model was not built by :func:`build_model`.
    """
    try:
        backbone = model.get_layer(BACKBONE_LAYER)
    except ValueError as error:
        raise ValueError(f"Model has no '{BACKBONE_LAYER}' layer.") from error
    if not isinstance(backbone, keras.Model):
        raise ValueError(f"Layer '{BACKBONE_LAYER}' is not a nested model.")
    return backbone


def unfreeze_top_layers(model: keras.Model, num_layers: int) -> int:
    """Make the final ``num_layers`` backbone layers trainable for fine-tuning.

    Batch-normalisation layers stay frozen: updating their statistics on a small
    medical dataset destabilises a pretrained backbone.

    Returns:
        The number of layers that became trainable.
    """
    if num_layers < 1:
        raise ValueError("num_layers must be positive.")
    backbone = get_backbone(model)
    backbone.trainable = True
    unfrozen = 0
    for index, layer in enumerate(backbone.layers):
        trainable = index >= len(backbone.layers) - num_layers and not isinstance(
            layer, keras.layers.BatchNormalization
        )
        layer.trainable = trainable
        unfrozen += int(trainable)
    LOGGER.info("Unfroze %d of the final %d backbone layers.", unfrozen, num_layers)
    return unfrozen
