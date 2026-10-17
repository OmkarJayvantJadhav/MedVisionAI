from __future__ import annotations

import numpy as np
import pytest
from medvision_ml.evaluation.robustness import (
    CORRUPTIONS,
    condition_metrics,
    gaussian_blur,
    jpeg,
    phone_like,
)

CLASSES = ["Melanoma", "Nevus"]


def _images(n: int = 3, size: int = 32) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(0, 255, (n, size, size, 3)).astype(np.float32)


def test_every_corruption_keeps_shape_type_and_range() -> None:
    images = _images()
    names = [c.name for c in CORRUPTIONS]
    assert names[0] == "clean" and len(set(names)) == len(names)
    for corruption in CORRUPTIONS:
        out = corruption.apply(images, np.random.default_rng(1))
        assert out.shape == images.shape and out.dtype == np.float32, corruption.name
        assert out.min() >= 0 and out.max() <= 255, corruption.name
    np.testing.assert_array_equal(CORRUPTIONS[0].apply(images, np.random.default_rng(1)), images)


def test_blur_keeps_flat_images_and_smooths_noise() -> None:
    flat = np.full((1, 16, 16, 3), 100.0, np.float32)
    np.testing.assert_allclose(gaussian_blur(flat, 2.0), flat, atol=1e-3)
    noisy = _images(1)
    assert gaussian_blur(noisy, 2.0).std() < 0.5 * noisy.std()


def test_jpeg_and_phone_change_the_image_a_little() -> None:
    images = gaussian_blur(_images(2, 64), 3.0)
    assert 0 < np.abs(jpeg(images, 90) - images).mean() < np.abs(jpeg(images, 10) - images).mean()
    phone = phone_like(images)
    assert phone[..., 2].mean() < images[..., 2].mean()  # warm cast lowers blue


def test_missed_malignant_counts_only_accepted_unflagged_images() -> None:
    # Rows: malignant flagged, malignant not flagged, malignant not flagged but
    # rejected by the gate, benign.
    logits = np.log(np.array([[0.9, 0.1], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9]]))
    labels = np.array([0, 0, 0, 1])
    result = condition_metrics(
        logits,
        labels,
        CLASSES,
        ["Melanoma"],
        temperature=1.0,
        malignant_threshold=0.5,
        accepted=[True, True, False, True],
    )
    assert result["malignant_recall"] == pytest.approx(1 / 3)
    assert result["malignant_missed"] == pytest.approx(1 / 3)
    assert result["accepted_by_gates"] == pytest.approx(0.75)
    assert result["malignant_precision"] == 1.0
    ungated = condition_metrics(
        logits, labels, CLASSES, ["Melanoma"], temperature=1.0, malignant_threshold=0.5
    )
    assert ungated["malignant_missed"] == pytest.approx(2 / 3)
