from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from app.inference.artifacts import load_artifacts
from app.inference.decision import RiskFlag, Status
from app.inference.errors import InvalidImageError, ModelArtifactError
from app.inference.predictor import Predictor
from PIL import Image

from tests.conftest import CLASSES, ImageFactory, write_model_dir


@pytest.fixture
def predictor(model_dir: Path) -> Predictor:
    return Predictor.from_directory(model_dir)


def test_red_image_is_confident_benign(predictor: Predictor, image_bytes: ImageFactory) -> None:
    result = predictor.predict(image_bytes((230, 40, 40)))
    assert result.predicted_class == "Acne"
    assert result.status is Status.OK and result.risk_flag is RiskFlag.BENIGN
    assert result.top_k[0].class_name == "Acne" and len(result.top_k) == 3
    assert sum(p.probability for p in result.top_k) <= 1.0
    assert result.model_version == "test-v0.0.1"
    assert result.calibrated is False and result.thresholds_fitted is False
    assert result.latency_ms > 0 and result.explanation is None


def test_green_image_is_flagged_malignant(predictor: Predictor, image_bytes: ImageFactory) -> None:
    result = predictor.predict(image_bytes((40, 230, 40)))
    assert result.predicted_class == "Melanoma"
    assert result.risk_flag is RiskFlag.MALIGNANT_SUSPECTED
    assert result.malignant_probability == pytest.approx(result.confidence)


def test_neutral_image_is_low_confidence(predictor: Predictor, image_bytes: ImageFactory) -> None:
    result = predictor.predict(image_bytes((128, 128, 128)))
    assert result.status is Status.LOW_CONFIDENCE


def test_dark_image_is_rejected(predictor: Predictor) -> None:
    buffer = io.BytesIO()
    Image.fromarray(np.full((64, 64, 3), 3, np.uint8)).save(buffer, format="PNG")
    result = predictor.predict(buffer.getvalue())
    assert result.status is Status.REJECTED and result.risk_flag is RiskFlag.UNCERTAIN


def test_invalid_upload_raises(predictor: Predictor) -> None:
    with pytest.raises(InvalidImageError):
        predictor.predict(b"nope")


def test_probabilities_match_onnx_softmax(predictor: Predictor, image_bytes: ImageFactory) -> None:
    from app.inference.preprocessing import decode_image, prepare_input

    data = image_bytes((90, 160, 200), size=(100, 70))
    onnx_probs, _ = predictor._run(prepare_input(decode_image(data), 32))
    result = predictor.predict(data)
    by_name = {p.class_name: p.probability for p in result.top_k}
    for name, value in by_name.items():
        assert value == pytest.approx(float(onnx_probs[0, CLASSES.index(name)]), abs=1e-6)


def test_temperature_scaling_is_applied(tmp_path: Path, image_bytes: ImageFactory) -> None:
    hot = Predictor.from_directory(
        write_model_dir(tmp_path / "hot", calibration={"temperature": 4.0, "fitted": True})
    )
    cold = Predictor.from_directory(write_model_dir(tmp_path / "cold"))
    data = image_bytes((230, 40, 40))
    assert hot.predict(data).confidence < cold.predict(data).confidence
    assert hot.predict(data).calibrated is True


def test_explanation_highlights_the_coloured_region(predictor: Predictor) -> None:
    array = np.full((64, 64, 3), 90, np.uint8)
    array += np.random.default_rng(0).integers(0, 20, array.shape, dtype=np.uint8)
    array[:32, :32] = (250, 30, 30)  # red patch in the top-left quadrant
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")

    result = predictor.predict(buffer.getvalue(), explain=True, explain_class="Acne")
    assert result.explanation is not None
    heatmap = result.explanation.heatmap
    assert heatmap.shape == (4, 4) and heatmap.max() == pytest.approx(1.0)
    assert heatmap[:2, :2].mean() > heatmap[2:, 2:].mean()
    overlay = Image.open(io.BytesIO(result.explanation.overlay_png))
    assert overlay.size == (64, 64)
    assert Image.open(io.BytesIO(result.explanation.heatmap_png)).size == (64, 64)


def test_explain_unknown_class(predictor: Predictor, image_bytes: ImageFactory) -> None:
    with pytest.raises(ValueError):
        predictor.predict(image_bytes(), explain=True, explain_class="Psoriasis")


def test_checksum_mismatch_is_detected(model_dir: Path) -> None:
    (model_dir / "head_weights.npz").write_bytes(b"tampered")
    with pytest.raises(ModelArtifactError, match="Checksum"):
        load_artifacts(model_dir)


def test_missing_files_are_detected(model_dir: Path) -> None:
    (model_dir / "model.onnx").unlink()
    with pytest.raises(ModelArtifactError, match="Missing"):
        load_artifacts(model_dir)


@pytest.mark.parametrize(
    "overrides",
    [
        {"format_version": 2},
        {"class_names": ["Only"]},
        {"malignant_classes": ["Unknown"]},
        {"calibration": {"temperature": 0, "fitted": True}},
        {"outputs": {"probabilities": "probabilities"}},
        {"class_names": [*CLASSES, "Extra"]},
    ],
)
def test_inconsistent_metadata_is_rejected(tmp_path: Path, overrides: dict[str, object]) -> None:
    directory = write_model_dir(tmp_path / "m", **overrides)
    with pytest.raises(ModelArtifactError):
        Predictor.from_directory(directory)


def test_graph_names_must_match_metadata(tmp_path: Path) -> None:
    directory = write_model_dir(
        tmp_path / "m", outputs={"probabilities": "probs", "feature_maps": "feature_maps"}
    )
    with pytest.raises(ModelArtifactError, match="do not match"):
        Predictor.from_directory(directory)


def test_invalid_json(model_dir: Path) -> None:
    (model_dir / "model_meta.json").write_text("{", encoding="utf-8")
    with pytest.raises(ModelArtifactError):
        load_artifacts(model_dir)
    (model_dir / "model_meta.json").unlink()
    with pytest.raises(ModelArtifactError):
        load_artifacts(model_dir)


def test_meta_round_trip_keeps_thresholds(model_dir: Path) -> None:
    raw = json.loads((model_dir / "model_meta.json").read_text(encoding="utf-8"))
    meta = load_artifacts(model_dir).meta
    assert meta.thresholds.malignant == raw["thresholds"]["malignant"]
    assert meta.malignant_indices == (1,)
