from __future__ import annotations

import numpy as np
import pytest
from medvision_ml.calibration.noise import noise_level


def _smooth(n: int = 2, size: int = 64) -> np.ndarray:
    y, x = np.mgrid[0:size, 0:size]
    ramp = (x + y)[None, ..., None] * 255.0 / (2 * size)
    return np.repeat(np.repeat(ramp, 3, axis=3), n, axis=0)


def test_smooth_gradients_have_no_noise() -> None:
    np.testing.assert_allclose(noise_level(_smooth()), 0.0, atol=1e-9)


@pytest.mark.parametrize("std", [2.0, 8.0, 20.0])
def test_estimate_recovers_added_gaussian_noise(std: float) -> None:
    rng = np.random.default_rng(0)
    noisy = _smooth(1, 224) + rng.normal(0, std, (1, 224, 224, 1))  # same noise in R, G, B
    # Keeping the quietest blocks reads about 0.73 of the true deviation for
    # pure Gaussian noise; the gate's threshold is fitted on this scale.
    assert noise_level(noisy)[0] == pytest.approx(0.73 * std, rel=0.05)


def test_local_texture_does_not_count_as_noise() -> None:
    image = _smooth(1, 128)
    image[0, :40, :40] += np.random.default_rng(1).normal(0, 40, (40, 40, 1))  # one hairy patch
    assert noise_level(image)[0] < 0.5


def test_tiny_images_are_refused() -> None:
    with pytest.raises(ValueError):
        noise_level(np.zeros((1, 9, 9, 3)))
