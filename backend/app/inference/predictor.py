"""The inference entry point: bytes in, a prediction (and optional explanation) out."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray

from app.inference.artifacts import ModelArtifacts, load_artifacts
from app.inference.calibration import energy_score, logits_from_features, softmax
from app.inference.cam import blend, cam_from_features, encode_png, render_heatmap
from app.inference.decision import Decision, Reason, RiskFlag, Status, decide
from app.inference.errors import ModelArtifactError
from app.inference.preprocessing import (
    DEFAULT_LIMITS,
    ImageLimits,
    decode_image,
    display_image,
    prepare_input,
)
from app.inference.quality import assess

TOP_K = 3


@dataclass(frozen=True)
class ClassProbability:
    class_name: str
    probability: float


@dataclass(frozen=True)
class Explanation:
    """Grad-CAM for the explained class, at feature-map resolution and as PNGs."""

    class_name: str
    heatmap: NDArray[np.float32]
    heatmap_png: bytes
    overlay_png: bytes


@dataclass(frozen=True)
class Prediction:
    status: Status
    predicted_class: str
    confidence: float
    top_k: tuple[ClassProbability, ...]
    risk_flag: RiskFlag
    malignant_probability: float
    reasons: tuple[Reason, ...]
    model_version: str
    calibrated: bool
    thresholds_fitted: bool
    latency_ms: float
    explanation: Explanation | None = None


class Predictor:
    """Holds one ONNX Runtime session; thread-safe for concurrent ``predict`` calls."""

    def __init__(
        self,
        artifacts: ModelArtifacts,
        limits: ImageLimits = DEFAULT_LIMITS,
        *,
        threads: int = 0,
    ) -> None:
        """Load the ONNX session.

        Args:
            artifacts: Verified model files and metadata.
            limits: Upload limits applied before decoding.
            threads: ONNX Runtime intra-op threads; 0 lets it use every visible
                core. On a CPU quota smaller than the visible cores (free hosting),
                1 avoids threads competing for the same slice of CPU.
        """
        self.artifacts = artifacts
        self.meta = artifacts.meta
        self.limits = limits
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = threads
        try:
            self.session = ort.InferenceSession(
                str(artifacts.onnx_path), options, providers=["CPUExecutionProvider"]
            )
        except Exception as error:  # ORT raises several unrelated exception types
            raise ModelArtifactError(f"Cannot load ONNX model: {error}") from error
        inputs = {i.name: i for i in self.session.get_inputs()}
        outputs = {o.name for o in self.session.get_outputs()}
        expected_outputs = {self.meta.probabilities_output, self.meta.feature_maps_output}
        if self.meta.input_name not in inputs or not expected_outputs <= outputs:
            raise ModelArtifactError("ONNX graph inputs/outputs do not match the metadata.")

    @classmethod
    def from_directory(
        cls, directory: Path, *, verify_checksums: bool = True, threads: int = 0
    ) -> Predictor:
        return cls(load_artifacts(directory, verify_checksums=verify_checksums), threads=threads)

    def _run(self, batch: NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        probabilities, feature_maps = self.session.run(
            [self.meta.probabilities_output, self.meta.feature_maps_output],
            {self.meta.input_name: batch},
        )
        return np.asarray(probabilities), np.asarray(feature_maps)

    def predict(
        self,
        data: bytes,
        *,
        explain: bool = False,
        explain_class: str | None = None,
        overlay_alpha: float = 0.4,
    ) -> Prediction:
        """Classify one uploaded image.

        Args:
            data: Raw file bytes (JPEG, PNG or WebP).
            explain: Also compute a Grad-CAM explanation.
            explain_class: Class to explain; defaults to the top prediction.
            overlay_alpha: Heatmap opacity in the overlay PNG.

        Raises:
            InvalidImageError: If the upload is not an acceptable image.
            ValueError: If ``explain_class`` is not a known class.
        """
        started = time.perf_counter()
        image = decode_image(data, self.limits)
        batch = prepare_input(image, self.meta.image_size)
        _, feature_maps = self._run(batch)

        logits = logits_from_features(feature_maps, self.artifacts.kernel, self.artifacts.bias)
        probabilities = softmax(logits, self.meta.temperature)[0]
        energy = float(energy_score(logits, self.meta.temperature)[0])
        decision: Decision = decide(
            probabilities,
            self.meta.malignant_indices,
            self.meta.thresholds,
            assess(batch[0]),
            energy,
        )

        order = np.argsort(probabilities)[::-1]
        top_k = tuple(
            ClassProbability(self.meta.class_names[i], float(probabilities[i]))
            for i in order[:TOP_K]
        )
        explanation = None
        if explain:
            target = int(order[0])
            if explain_class is not None:
                if explain_class not in self.meta.class_names:
                    raise ValueError(f"Unknown class: {explain_class}")
                target = self.meta.class_names.index(explain_class)
            explanation = self._explain(image, feature_maps[0], target, overlay_alpha)

        return Prediction(
            status=decision.status,
            predicted_class=top_k[0].class_name,
            confidence=top_k[0].probability,
            top_k=top_k,
            risk_flag=decision.risk_flag,
            malignant_probability=decision.malignant_probability,
            reasons=decision.reasons,
            model_version=self.meta.model_version,
            calibrated=self.meta.calibration_fitted,
            thresholds_fitted=self.meta.thresholds.fitted,
            latency_ms=(time.perf_counter() - started) * 1000,
            explanation=explanation,
        )

    def _explain(
        self, image: NDArray[np.uint8], feature_maps: NDArray[np.float32], target: int, alpha: float
    ) -> Explanation:
        heatmap = cam_from_features(feature_maps, self.artifacts.kernel, target)
        shown = display_image(image)
        coloured = render_heatmap(heatmap, shown.shape[0], shown.shape[1])
        return Explanation(
            class_name=self.meta.class_names[target],
            heatmap=heatmap,
            heatmap_png=encode_png(coloured),
            overlay_png=encode_png(blend(shown, coloured, alpha)),
        )
