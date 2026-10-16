"""Matplotlib figures for training and evaluation reports."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from medvision_ml.evaluation.metrics import ReliabilityBins


def _save(figure: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_training_history(history: Mapping[str, Sequence[float]], path: Path) -> Path:
    """Plot loss and accuracy curves for training and validation."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, metric in zip(axes, ("loss", "accuracy"), strict=True):
        for prefix, label in (("", "train"), ("val_", "validation")):
            values = history.get(f"{prefix}{metric}")
            if values:
                axis.plot(range(1, len(values) + 1), values, label=label)
        axis.set_title(metric.title())
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.3)
        axis.legend()
    return _save(figure, path)


def plot_confusion_matrix(
    matrix: Sequence[Sequence[int]], class_names: Sequence[str], path: Path
) -> Path:
    """Plot a row-normalised confusion matrix annotated with raw counts."""
    counts = np.asarray(matrix, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        normalised = np.nan_to_num(counts / counts.sum(axis=1, keepdims=True))
    size = max(6.0, 0.8 * len(class_names))
    figure, axis = plt.subplots(figsize=(size + 1, size))
    image = axis.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    figure.colorbar(image, ax=axis, label="Share of true class")
    ticks = range(len(class_names))
    axis.set_xticks(ticks, class_names, rotation=45, ha="right")
    axis.set_yticks(ticks, class_names)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    for i in ticks:
        for j in ticks:
            colour = "white" if normalised[i, j] > 0.5 else "black"
            axis.text(
                j, i, f"{int(counts[i, j])}", ha="center", va="center", color=colour, fontsize=8
            )
    return _save(figure, path)


def plot_reliability_diagram(bins: ReliabilityBins, ece: float, path: Path) -> Path:
    """Plot accuracy against confidence per bin, with the perfect-calibration diagonal."""
    figure, axis = plt.subplots(figsize=(5, 5))
    centres = (bins.edges[:-1] + bins.edges[1:]) / 2
    width = bins.edges[1] - bins.edges[0]
    filled = bins.counts > 0
    axis.bar(centres[filled], bins.accuracy[filled], width=width * 0.9, alpha=0.7, label="Accuracy")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Confidence")
    axis.set_ylabel("Accuracy")
    axis.set_title(f"Reliability (ECE = {ece:.3f})")
    axis.legend(loc="upper left")
    return _save(figure, path)
