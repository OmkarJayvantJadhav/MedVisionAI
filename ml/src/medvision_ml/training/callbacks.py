"""Standard Keras callbacks for one training stage."""

from __future__ import annotations

from pathlib import Path

import keras

from medvision_ml.config import TrainingConfig


def build_callbacks(
    config: TrainingConfig,
    checkpoint_path: Path,
    history_csv: Path,
) -> list[keras.callbacks.Callback]:
    """Early stopping, best-checkpoint saving, LR reduction and CSV logging on ``val_loss``."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path), monitor="val_loss", save_best_only=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.reduce_lr_factor,
            patience=config.reduce_lr_patience,
            min_lr=config.min_learning_rate,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(history_csv), append=True),
    ]
