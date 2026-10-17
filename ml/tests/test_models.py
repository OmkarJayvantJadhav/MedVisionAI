"""Model construction and Grad-CAM parity, with random weights (no downloads)."""

from __future__ import annotations

from pathlib import Path

import keras
import numpy as np
import pytest
from medvision_ml.config import AugmentationConfig, ModelConfig
from medvision_ml.explainability.gradcam import cam_from_features, gradcam, head_weights, overlay
from medvision_ml.models.builder import (
    BACKBONE_LAYER,
    LOGITS_LAYER,
    PROBABILITIES_LAYER,
    build_model,
    get_backbone,
    unfreeze_top_layers,
)
from medvision_ml.models.io import load_class_names, load_model, save_model

SIZE = 64
CLASSES = 5


@pytest.fixture(scope="module")
def model() -> keras.Model:
    keras.utils.set_random_seed(0)
    return build_model(ModelConfig(), AugmentationConfig(), CLASSES, SIZE, weights=None)


@pytest.fixture
def image() -> np.ndarray:
    return np.random.default_rng(0).uniform(0, 255, (SIZE, SIZE, 3)).astype(np.float32)


def test_model_layout_and_output(model: keras.Model, image: np.ndarray) -> None:
    names = [layer.name for layer in model.layers]
    assert names[-3:] == ["dropout", LOGITS_LAYER, PROBABILITIES_LAYER]
    assert BACKBONE_LAYER in names
    assert not get_backbone(model).trainable
    probs = np.asarray(model(image[None], training=False))
    assert probs.shape == (1, CLASSES)
    assert probs.sum() == pytest.approx(1.0, abs=1e-5)


def test_mobilenet_rescales_raw_pixels() -> None:
    model = build_model(
        ModelConfig(backbone="mobilenetv2"), AugmentationConfig(), 3, 96, weights=None
    )
    assert "input_scaling" in [layer.name for layer in model.layers]


def test_unfreeze_keeps_batch_norm_frozen(model: keras.Model) -> None:
    clone = keras.models.clone_model(model)
    unfrozen = unfreeze_top_layers(clone, 20)
    backbone = get_backbone(clone)
    assert 0 < unfrozen <= 20
    for layer in backbone.layers[-20:]:
        if isinstance(layer, keras.layers.BatchNormalization):
            assert not layer.trainable
    assert not any(layer.trainable for layer in backbone.layers[:-20])


def test_build_rejects_single_class() -> None:
    with pytest.raises(ValueError):
        build_model(ModelConfig(), AugmentationConfig(), 1, SIZE, weights=None)


def test_numpy_cam_matches_tensorflow_gradcam(model: keras.Model, image: np.ndarray) -> None:
    kernel, _ = head_weights(model)
    probs = np.asarray(model(image[None], training=False))[0]
    for class_index in range(CLASSES):
        result = gradcam(model, image, class_index)
        np.testing.assert_allclose(result.probabilities, probs, atol=1e-5)
        cam = cam_from_features(result.feature_maps, kernel, class_index)
        if not cam.any() and not result.heatmap.any():
            continue
        cosine = (
            cam.ravel()
            @ result.heatmap.ravel()
            / (np.linalg.norm(cam) * np.linalg.norm(result.heatmap))
        )
        assert cosine >= 0.999


def test_gradcam_defaults_to_top_class_and_overlays(model: keras.Model, image: np.ndarray) -> None:
    result = gradcam(model, image)
    assert result.class_index == int(result.probabilities.argmax())
    assert result.heatmap.min() >= 0.0 and result.heatmap.max() <= 1.0
    blended = overlay(image, result.heatmap)
    assert blended.shape == image.shape and blended.dtype == np.uint8


def test_save_and_load_round_trip_without_unsafe_mode(
    model: keras.Model, image: np.ndarray, tmp_path: Path
) -> None:
    names = [f"class_{i}" for i in range(CLASSES)]
    names_path = save_model(model, tmp_path / "model.keras", names)
    restored = load_model(tmp_path / "model.keras")
    assert load_class_names(names_path) == names
    np.testing.assert_allclose(
        np.asarray(restored(image[None], training=False)),
        np.asarray(model(image[None], training=False)),
        atol=1e-6,
    )


def test_save_requires_keras_suffix(model: keras.Model, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        save_model(model, tmp_path / "model.h5", ["a", "b"])


def test_load_class_names_validates(tmp_path: Path) -> None:
    path = tmp_path / "names.json"
    path.write_text('["only"]', encoding="utf-8")
    with pytest.raises(ValueError):
        load_class_names(path)


def test_mixed_precision_model_loads_and_exports_in_float32(tmp_path: Path) -> None:
    """A model trained with mixed precision must be evaluated and exported in float32.

    Regression: exporting the first ISIC model failed its parity check (3.5e-3)
    because Keras computed the reference probabilities in float16.
    """
    from medvision_ml.export.onnx_export import export_model

    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        keras.utils.set_random_seed(1)
        mixed = build_model(ModelConfig(), AugmentationConfig(), CLASSES, SIZE, weights=None)
        save_model(mixed, tmp_path / "mixed.keras", [f"c{i}" for i in range(CLASSES)])
    finally:
        keras.mixed_precision.set_global_policy("float32")

    restored = load_model(tmp_path / "mixed.keras")
    policies = {layer.dtype_policy.name for layer in restored.layers}
    assert policies == {"float32"}
    for original, copy in zip(mixed.weights, restored.weights, strict=True):
        np.testing.assert_array_equal(np.asarray(original), np.asarray(copy))

    result = export_model(
        restored,
        tmp_path / "export",
        class_names=[f"c{i}" for i in range(CLASSES)],
        malignant_classes=[],
        model_version="mixed-test",
        backbone="efficientnetb0",
        image_size=SIZE,
    )
    assert result.max_probability_diff <= 1e-4


def test_float32_reload_keeps_input_normalisation(tmp_path: Path) -> None:
    """Rebuilding a mixed-precision model in float32 must keep every layer's behaviour.

    Regression (D40): weights were copied with ``set_weights``, which skipped the
    Normalization layer's load hook, so the reloaded model stopped normalising its
    input. Evaluation and export then scored the first ISIC model ~5 points lower
    than training did. Untrained ``weights=None`` models hid this because their
    normalisation statistics are the identity.
    """
    keras.mixed_precision.set_global_policy("mixed_float16")
    try:
        keras.utils.set_random_seed(2)
        mixed = build_model(ModelConfig(), AugmentationConfig(), CLASSES, SIZE, weights=None)
        normalisation = next(
            layer
            for layer in get_backbone(mixed).layers
            if isinstance(layer, keras.layers.Normalization)
        )
        rng = np.random.default_rng(0)
        normalisation.adapt(rng.normal(0.6, 0.05, size=(16, SIZE, SIZE, 3)).astype("float32"))
        save_model(mixed, tmp_path / "mixed.keras", [f"c{i}" for i in range(CLASSES)])
        images = rng.uniform(0, 255, size=(4, SIZE, SIZE, 3)).astype("float32")
        expected = np.asarray(get_backbone(mixed)(images, training=False), dtype=np.float32)
    finally:
        keras.mixed_precision.set_global_policy("float32")

    restored = load_model(tmp_path / "mixed.keras")
    actual = np.asarray(get_backbone(restored)(images, training=False))
    relative = np.abs(actual - expected).max() / np.abs(expected).max()
    assert relative < 0.02
