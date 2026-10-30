"""Training losses, learning-rate schedules and the callbacks that go with them."""

from __future__ import annotations

from pathlib import Path

import keras
import numpy as np
import pytest
from medvision_ml.config import TrainingConfig
from medvision_ml.training.callbacks import build_callbacks
from medvision_ml.training.losses import (
    SparseFocalLoss,
    build_loss,
    learning_rate,
    steps_per_epoch,
)

LABELS = np.array([0, 1, 2, 1])
PROBS = np.array(
    [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4], [0.6, 0.3, 0.1]], dtype="float32"
)


def test_focal_loss_with_zero_gamma_equals_cross_entropy() -> None:
    focal = float(SparseFocalLoss(gamma=0.0)(LABELS, PROBS))
    crossentropy = float(keras.losses.SparseCategoricalCrossentropy()(LABELS, PROBS))
    assert focal == pytest.approx(crossentropy, rel=1e-5)


def test_focal_loss_down_weights_confident_correct_examples() -> None:
    loss = SparseFocalLoss(gamma=2.0, reduction=None)
    per_example = np.asarray(loss(LABELS, PROBS))
    crossentropy = -np.log(PROBS[np.arange(4), LABELS])
    expected = (1 - PROBS[np.arange(4), LABELS]) ** 2 * crossentropy
    np.testing.assert_allclose(per_example, expected, rtol=1e-5)
    # The confident correct example (p = 0.8) is reduced far more than the hard one (p = 0.3).
    assert per_example[1] / crossentropy[1] < per_example[3] / crossentropy[3]


def test_focal_loss_round_trips_through_keras_serialisation() -> None:
    restored = keras.losses.deserialize(keras.losses.serialize(SparseFocalLoss(gamma=1.5)))
    assert isinstance(restored, SparseFocalLoss)
    assert restored.gamma == 1.5
    with pytest.raises(ValueError):
        SparseFocalLoss(gamma=-1)


def test_build_loss_follows_the_configuration() -> None:
    assert isinstance(build_loss(TrainingConfig()), keras.losses.SparseCategoricalCrossentropy)
    focal = build_loss(TrainingConfig(loss="focal", focal_gamma=3.0))
    assert isinstance(focal, SparseFocalLoss) and focal.gamma == 3.0


def test_plateau_schedule_keeps_a_constant_rate() -> None:
    assert learning_rate(TrainingConfig(), 1e-4, epochs=10, steps_per_epoch=5) == 1e-4


def test_cosine_schedule_warms_up_then_decays_to_zero() -> None:
    config = TrainingConfig(lr_schedule="cosine", warmup_epochs=2)
    schedule = learning_rate(config, 1e-3, epochs=10, steps_per_epoch=5)
    assert isinstance(schedule, keras.optimizers.schedules.LearningRateSchedule)
    assert float(schedule(0)) == pytest.approx(0.0, abs=1e-9)
    assert float(schedule(5)) == pytest.approx(5e-4, rel=1e-4)
    assert float(schedule(10)) == pytest.approx(1e-3, rel=1e-4)
    assert float(schedule(30)) < 1e-3
    assert float(schedule(50)) == pytest.approx(0.0, abs=1e-8)


def test_cosine_schedule_without_warmup_starts_at_the_base_rate() -> None:
    schedule = learning_rate(TrainingConfig(lr_schedule="cosine"), 1e-3, 4, 10)
    assert float(schedule(0)) == pytest.approx(1e-3, rel=1e-6)
    assert float(schedule(40)) == pytest.approx(0.0, abs=1e-8)


def test_steps_per_epoch_rounds_up() -> None:
    assert steps_per_epoch(65, 32) == 3
    assert steps_per_epoch(0, 32) == 1


@pytest.mark.parametrize(("schedule", "reduces"), [("plateau", True), ("cosine", False)])
def test_learning_rate_reduction_only_with_the_plateau_schedule(
    tmp_path: Path, schedule: str, reduces: bool
) -> None:
    callbacks = build_callbacks(
        TrainingConfig(lr_schedule=schedule),  # type: ignore[arg-type]
        tmp_path / "ckpt.keras",
        tmp_path / "history.csv",
    )
    kinds = [type(c) for c in callbacks]
    assert (keras.callbacks.ReduceLROnPlateau in kinds) is reduces
    assert kinds[-1] is keras.callbacks.CSVLogger
