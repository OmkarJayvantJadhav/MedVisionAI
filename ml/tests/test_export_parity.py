"""Parity between the TensorFlow pipeline and the TensorFlow-free serving code.

Imports the backend's inference core (NumPy + Pillow + ONNX Runtime) and checks
it against Keras, TensorFlow Grad-CAM and the training input pipeline.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import keras
import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

BACKEND = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(BACKEND))

from app.inference.calibration import mahalanobis_distance, pooled_features  # noqa: E402
from app.inference.cam import cam_from_features as serving_cam  # noqa: E402
from app.inference.predictor import Predictor  # noqa: E402
from app.inference.preprocessing import decode_image, prepare_input  # noqa: E402
from app.inference.quality import grayscale  # noqa: E402
from app.inference.quality import noise_level as serving_noise  # noqa: E402
from medvision_ml.calibration.noise import noise_level  # noqa: E402
from medvision_ml.calibration.ood import fit_gaussian, mahalanobis  # noqa: E402
from medvision_ml.config import AugmentationConfig, ModelConfig  # noqa: E402
from medvision_ml.explainability.gradcam import gradcam  # noqa: E402
from medvision_ml.export.onnx_export import Thresholds, export_model, sha256_file  # noqa: E402
from medvision_ml.models.builder import POOL_LAYER, build_model  # noqa: E402

SIZE = 64
CLASSES = ["Acne", "Melanoma", "Nevus"]


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> tuple[keras.Model, Path]:
    keras.utils.set_random_seed(3)
    model = build_model(ModelConfig(), AugmentationConfig(), len(CLASSES), SIZE, weights=None)
    directory = tmp_path_factory.mktemp("export")
    # Out-of-distribution statistics fitted on random images; the threshold is
    # high enough that the gate never rejects the test images.
    pooled = pool_model(model)(np.random.default_rng(5).uniform(0, 255, (16, SIZE, SIZE, 3)))
    stats = tmp_path_factory.mktemp("ood") / "ood_stats.npz"
    mean, precision = fit_gaussian(np.asarray(pooled), shrinkage=0.1)
    np.savez(stats, mean=mean.astype(np.float32), precision=precision.astype(np.float32))
    export_model(
        model,
        directory,
        class_names=CLASSES,
        malignant_classes=["Melanoma"],
        model_version="test-v1",
        backbone="efficientnetb0",
        image_size=SIZE,
        thresholds=Thresholds(min_sharpness=0.0, ood_mahalanobis=1e12),
        ood_stats=stats,
    )
    return model, directory


def pool_model(model: keras.Model) -> keras.Model:
    return keras.Model(model.input, model.get_layer(POOL_LAYER).output)


def photo(seed: int, size: tuple[int, int]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    smooth = np.asarray(Image.fromarray(coarse).resize(size, Image.Resampling.BICUBIC))
    noise = rng.integers(-10, 11, smooth.shape)
    return np.clip(smooth.astype(int) + noise, 0, 255).astype(np.uint8)


def encode(array: np.ndarray, fmt: str) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format=fmt, quality=95)
    return buffer.getvalue()


def test_metadata_and_checksums(exported: tuple[keras.Model, Path]) -> None:
    _, directory = exported
    meta = json.loads((directory / "model_meta.json").read_text())
    assert meta["class_names"] == CLASSES and meta["malignant_classes"] == ["Melanoma"]
    assert meta["parity"]["max_probability_diff"] <= 1e-4
    for name, entry in meta["files"].items():
        assert entry["sha256"] == sha256_file(directory / name)


# PNG is lossless, so any difference would come from resizing. JPEG differs
# slightly more because TensorFlow decodes with the fast integer DCT by default
# while Pillow uses the accurate one (about one grey level on average).
@pytest.mark.parametrize(("fmt", "tolerance"), [("PNG", 1e-3), ("JPEG", 2.0)])
def test_preprocessing_matches_training_pipeline(
    tmp_path: Path, fmt: str, tolerance: float
) -> None:
    folder = tmp_path / "split" / "only"
    folder.mkdir(parents=True)
    arrays = [photo(seed, size) for seed, size in enumerate([(300, 200), (97, 131), (64, 64)])]
    for i, array in enumerate(arrays):
        (folder / f"{i}.{fmt.lower()}").write_bytes(encode(array, fmt))
    dataset = keras.utils.image_dataset_from_directory(
        tmp_path / "split", image_size=(SIZE, SIZE), batch_size=8, shuffle=False
    )
    expected = next(iter(dataset))[0].numpy()
    for i, path in enumerate(dataset.file_paths):
        served = prepare_input(decode_image(Path(path).read_bytes()), SIZE)[0]
        assert np.abs(served - expected[i]).mean() <= tolerance


def test_onnx_probabilities_match_keras(exported: tuple[keras.Model, Path]) -> None:
    model, directory = exported
    predictor = Predictor.from_directory(directory)
    batch = np.random.default_rng(0).uniform(0, 255, (3, SIZE, SIZE, 3)).astype(np.float32)
    onnx_probs, _ = predictor._run(batch)
    np.testing.assert_allclose(onnx_probs, model(batch, training=False), atol=1e-4)


def test_export_ignores_every_training_augmentation(tmp_path: Path) -> None:
    # Augmentation layers are part of the model but must be inactive in the
    # exported inference graph, whichever options a training run switches on.
    keras.utils.set_random_seed(4)
    augmentation = AugmentationConfig(
        horizontal_flip=True,
        vertical_flip=True,
        rotation=0.5,
        zoom=0.2,
        contrast=0.2,
        brightness=0.2,
        saturation=0.3,
    )
    model = build_model(ModelConfig(), augmentation, len(CLASSES), SIZE, weights=None)
    export_model(
        model,
        tmp_path,
        class_names=CLASSES,
        malignant_classes=["Melanoma"],
        model_version="test-aug",
        backbone="efficientnetb0",
        image_size=SIZE,
        thresholds=Thresholds(min_sharpness=0.0),
    )
    batch = np.random.default_rng(1).uniform(0, 255, (3, SIZE, SIZE, 3)).astype(np.float32)
    onnx_probs, _ = Predictor.from_directory(tmp_path)._run(batch)
    np.testing.assert_allclose(onnx_probs, model(batch, training=False), atol=1e-4)


def test_serving_cam_matches_tensorflow_gradcam(exported: tuple[keras.Model, Path]) -> None:
    model, directory = exported
    predictor = Predictor.from_directory(directory)
    image = photo(7, (SIZE, SIZE)).astype(np.float32)
    _, feature_maps = predictor._run(image[np.newaxis])
    for class_index in range(len(CLASSES)):
        reference = gradcam(model, image, class_index).heatmap
        served = serving_cam(feature_maps[0], predictor.artifacts.kernel, class_index)
        if not reference.any() and not served.any():
            continue
        cosine = (
            served.ravel()
            @ reference.ravel()
            / (np.linalg.norm(served) * np.linalg.norm(reference))
        )
        assert cosine >= 0.99


def test_end_to_end_prediction_matches_keras(exported: tuple[keras.Model, Path]) -> None:
    model, directory = exported
    data = encode(photo(11, (120, 90)), "PNG")
    result = Predictor.from_directory(directory).predict(data, explain=True)
    image = tf.image.resize(tf.io.decode_png(data, channels=3), (SIZE, SIZE)).numpy()
    expected = np.asarray(model(image[np.newaxis], training=False))[0]
    assert result.predicted_class == CLASSES[int(expected.argmax())]
    assert result.confidence == pytest.approx(float(expected.max()), abs=1e-4)
    assert result.explanation is not None


def test_serving_ood_distance_matches_training_pipeline(
    exported: tuple[keras.Model, Path],
) -> None:
    model, directory = exported
    predictor = Predictor.from_directory(directory)
    artifacts = predictor.artifacts
    assert artifacts.ood_mean is not None and artifacts.ood_precision is not None
    batch = np.stack([photo(seed, (SIZE, SIZE)) for seed in (21, 22, 23)]).astype(np.float32)
    _, feature_maps = predictor._run(batch)
    served = mahalanobis_distance(
        pooled_features(feature_maps), artifacts.ood_mean, artifacts.ood_precision
    )
    expected = mahalanobis(
        np.asarray(pool_model(model)(batch, training=False)),
        artifacts.ood_mean,
        artifacts.ood_precision,
    )
    np.testing.assert_allclose(served, expected, rtol=1e-3)


def test_serving_noise_estimate_matches_training_pipeline() -> None:
    rng = np.random.default_rng(4)
    batch = np.stack(
        [
            photo(seed, (SIZE, SIZE)) + rng.normal(0, std, (SIZE, SIZE, 3))
            for seed, std in ((31, 0.0), (32, 5.0), (33, 20.0))
        ]
    )
    served = [serving_noise(grayscale(image)) for image in batch]
    np.testing.assert_allclose(served, noise_level(batch), rtol=1e-9)
