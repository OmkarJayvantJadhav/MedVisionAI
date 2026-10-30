"""Training losses and learning-rate schedules."""

from __future__ import annotations

import math
from typing import Any

import keras

from medvision_ml.config import TrainingConfig


@keras.saving.register_keras_serializable(package="medvision")
class SparseFocalLoss(keras.losses.Loss):  # type: ignore[misc]  # Keras is untyped
    """Focal loss for integer labels and softmax probabilities.

    ``-(1 - p_t) ** gamma * log(p_t)``, where ``p_t`` is the probability of the
    true class. Confident, correct examples contribute little, so training
    concentrates on hard and rare ones. ``gamma = 0`` is ordinary sparse
    categorical cross-entropy.
    """

    def __init__(self, gamma: float = 2.0, name: str = "sparse_focal_loss", **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)
        if gamma < 0:
            raise ValueError("gamma must be non-negative.")
        self.gamma = float(gamma)

    def call(self, y_true: Any, y_pred: Any) -> Any:
        y_pred = keras.ops.convert_to_tensor(y_pred, dtype="float32")
        labels = keras.ops.reshape(keras.ops.cast(y_true, "int32"), (-1,))
        one_hot = keras.ops.one_hot(labels, keras.ops.shape(y_pred)[-1])
        eps = keras.config.epsilon()
        p_true = keras.ops.clip(keras.ops.sum(one_hot * y_pred, axis=-1), eps, 1.0 - eps)
        return -keras.ops.power(1.0 - p_true, self.gamma) * keras.ops.log(p_true)

    def get_config(self) -> dict[str, Any]:
        return {**super().get_config(), "gamma": self.gamma}


def build_loss(config: TrainingConfig) -> keras.losses.Loss:
    """Return the loss named in the configuration."""
    if config.loss == "focal":
        return SparseFocalLoss(config.focal_gamma)
    return keras.losses.SparseCategoricalCrossentropy()


def learning_rate(
    config: TrainingConfig, base_rate: float, epochs: int, steps_per_epoch: int
) -> float | keras.optimizers.schedules.LearningRateSchedule:
    """Return the learning rate for one training stage.

    With the ``plateau`` schedule this is the constant ``base_rate`` (the
    ReduceLROnPlateau callback lowers it). With ``cosine`` the rate rises
    linearly from zero over ``warmup_epochs`` and then follows a cosine decay
    to zero by the end of the stage.
    """
    if config.lr_schedule == "plateau":
        return base_rate
    steps_per_epoch = max(1, steps_per_epoch)
    warmup_steps = min(config.warmup_epochs, epochs) * steps_per_epoch
    decay_steps = max(1, epochs * steps_per_epoch - warmup_steps)
    if warmup_steps == 0:
        return keras.optimizers.schedules.CosineDecay(base_rate, decay_steps)
    return keras.optimizers.schedules.CosineDecay(
        0.0, decay_steps, warmup_target=base_rate, warmup_steps=warmup_steps
    )


def steps_per_epoch(num_images: int, batch_size: int) -> int:
    """Number of batches in one pass over ``num_images`` images."""
    return max(1, math.ceil(num_images / batch_size))
