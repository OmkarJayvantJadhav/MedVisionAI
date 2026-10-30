"""Two-stage transfer-learning training.

Stage 1 trains only the classification head on a frozen backbone. Stage 2
unfreezes the top of the backbone (or all of it) and fine-tunes with a lower
learning rate. The loss and the learning-rate schedule come from the
configuration. The final model keeps the weights with the lowest validation
loss across both stages.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import keras
import tensorflow as tf

from medvision_ml.config import ProjectConfig
from medvision_ml.data.dataset import class_counts, load_split
from medvision_ml.data.images import SPLITS
from medvision_ml.evaluation.plots import plot_training_history
from medvision_ml.models.builder import build_model, unfreeze_top_layers
from medvision_ml.models.io import save_model
from medvision_ml.training.callbacks import build_callbacks
from medvision_ml.training.class_weights import compute_class_weights
from medvision_ml.training.losses import build_loss, learning_rate, steps_per_epoch
from medvision_ml.training.reproducibility import set_seed

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingResult:
    """Paths and summary values produced by :func:`train`."""

    run_name: str
    model_path: Path
    class_names_path: Path
    run_dir: Path
    best_val_loss: float
    best_val_accuracy: float
    epochs_run: int
    duration_seconds: float


def default_run_name(config: ProjectConfig) -> str:
    """Return ``<backbone>-<UTC timestamp>``."""
    return f"{config.model.backbone}-{datetime.now(UTC):%Y%m%d-%H%M%S}"


def _configure_precision(enabled: bool) -> None:
    # Mixed precision only pays off on GPUs; on CPUs it is slower.
    use_mixed = enabled and bool(tf.config.list_physical_devices("GPU"))
    keras.mixed_precision.set_global_policy("mixed_float16" if use_mixed else "float32")
    LOGGER.info("Precision policy: %s", keras.mixed_precision.global_policy().name)


def _compile(
    model: keras.Model,
    config: ProjectConfig,
    base_rate: float,
    epochs: int,
    steps: int,
) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate(config.training, base_rate, epochs, steps)
        ),
        loss=build_loss(config.training),
        metrics=["accuracy"],
    )


def train(config: ProjectConfig, run_name: str | None = None) -> TrainingResult:
    """Train a classifier on ``config.data_dir`` and save it under ``config.model_dir``.

    Raises:
        FileNotFoundError: If a dataset split is missing.
        ValueError: If the train and validation class folders differ.
    """
    run_name = run_name or default_run_name(config)
    run_dir = config.output_dir / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    model_path = config.model_dir / run_name / "model.keras"
    train_dir, val_dir = (config.data_dir / s for s in SPLITS[:2])

    set_seed(config.seed)
    _configure_precision(config.training.mixed_precision)

    train_split = load_split(train_dir, config.data, shuffle=True, seed=config.seed)
    val_split = load_split(val_dir, config.data, shuffle=False, seed=config.seed)
    class_names = train_split.class_names
    if class_names != val_split.class_names:
        raise ValueError(
            f"Class folders differ: train={class_names} validation={val_split.class_names}"
        )
    train_ds, val_ds = train_split.dataset, val_split.dataset

    counts = class_counts(train_dir)
    steps = steps_per_epoch(sum(counts.values()), config.data.batch_size)
    class_weight = compute_class_weights(
        {i: counts[name] for i, name in enumerate(class_names)}, config.training.class_weighting
    )
    (run_dir / "config.json").write_text(
        json.dumps(
            {"config": asdict(config), "class_names": class_names, "class_weight": class_weight},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    model = build_model(
        config.model,
        config.augmentation,
        len(class_names),
        config.data.image_size,
        weights="imagenet" if config.model.pretrained else None,
    )
    history_csv = run_dir / "history.csv"
    checkpoint = run_dir / "checkpoint.keras"
    histories: list[dict[str, list[float]]] = []
    started = time.perf_counter()

    LOGGER.info("Stage 1: training the head for up to %d epochs.", config.training.epochs)
    _compile(model, config, config.training.learning_rate, config.training.epochs, steps)
    stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.training.epochs,
        class_weight=class_weight,
        shuffle=False,  # the tf.data pipeline already shuffles
        callbacks=build_callbacks(config.training, checkpoint, history_csv),
    )
    histories.append(stage1.history)
    stage1_best = min(stage1.history["val_loss"])
    stage1_weights = model.get_weights()

    if config.training.fine_tune_epochs > 0:
        LOGGER.info("Stage 2: fine-tuning for up to %d epochs.", config.training.fine_tune_epochs)
        unfreeze_top_layers(model, config.training.fine_tune_layers)
        _compile(
            model,
            config,
            config.training.fine_tune_learning_rate,
            config.training.fine_tune_epochs,
            steps,
        )
        initial_epoch = len(stage1.history["loss"])
        stage2 = model.fit(
            train_ds,
            validation_data=val_ds,
            initial_epoch=initial_epoch,
            epochs=initial_epoch + config.training.fine_tune_epochs,
            class_weight=class_weight,
            shuffle=False,  # the tf.data pipeline already shuffles
            callbacks=build_callbacks(config.training, checkpoint, history_csv),
        )
        histories.append(stage2.history)
        if min(stage2.history["val_loss"]) >= stage1_best:
            LOGGER.info("Fine-tuning did not improve validation loss; keeping stage-1 weights.")
            model.set_weights(stage1_weights)

    duration = time.perf_counter() - started
    merged: dict[str, list[float]] = {}
    for history in histories:
        for key, values in history.items():
            merged.setdefault(key, []).extend(float(v) for v in values)

    names_path = save_model(model, model_path, class_names)
    plot_training_history(merged, run_dir / "history.png")
    best_epoch = min(range(len(merged["val_loss"])), key=merged["val_loss"].__getitem__)
    result = TrainingResult(
        run_name=run_name,
        model_path=model_path,
        class_names_path=names_path,
        run_dir=run_dir,
        best_val_loss=merged["val_loss"][best_epoch],
        best_val_accuracy=merged["val_accuracy"][best_epoch],
        epochs_run=len(merged["val_loss"]),
        duration_seconds=duration,
    )
    summary: dict[str, Any] = {
        k: str(v) if isinstance(v, Path) else v for k, v in asdict(result).items()
    }
    (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOGGER.info("Training finished: %s", summary)
    return result
