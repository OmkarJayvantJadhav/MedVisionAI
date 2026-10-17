from __future__ import annotations

import json
from pathlib import Path

import pytest
from app.inference.artifacts import load_artifacts
from app.inference.download import download_model, main
from app.inference.errors import ModelArtifactError

from tests.conftest import write_model_dir


@pytest.fixture
def release(tmp_path: Path) -> str:
    return write_model_dir(tmp_path / "release").as_uri()


def test_download_verifies_and_installs(release: str, tmp_path: Path) -> None:
    meta = download_model(release, tmp_path / "models")
    assert meta.model_version == "test-v0.0.1"
    assert load_artifacts(tmp_path / "models").meta == meta


def test_second_download_skips_valid_files(release: str, tmp_path: Path) -> None:
    dest = tmp_path / "models"
    download_model(release, dest)
    before = (dest / "model.onnx").stat().st_mtime_ns
    download_model(release, dest)
    assert (dest / "model.onnx").stat().st_mtime_ns == before


def test_tampered_file_is_rejected_and_not_installed(tmp_path: Path) -> None:
    source = write_model_dir(tmp_path / "release")
    (source / "model.onnx").write_bytes(b"tampered")
    dest = tmp_path / "models"
    with pytest.raises(ModelArtifactError, match="Checksum"):
        download_model(source.as_uri(), dest)
    assert not (dest / "model.onnx").exists()
    assert not (dest / "model_meta.json").exists()
    assert not list(dest.glob(".*"))  # no temporary files left behind


def test_unsupported_scheme(tmp_path: Path) -> None:
    with pytest.raises(ModelArtifactError, match="scheme"):
        download_model("ftp://example.com/models", tmp_path)


def test_cli_exit_codes(release: str, tmp_path: Path) -> None:
    assert main(["--base-url", release, "--dest", str(tmp_path / "ok")]) == 0
    assert main(["--base-url", (tmp_path / "missing").as_uri(), "--dest", str(tmp_path / "x")]) == 1


def test_every_listed_file_is_downloaded(tmp_path: Path) -> None:
    # Regression: only model.onnx and head_weights.npz were fetched, so a model
    # with an out-of-distribution gate failed to load after download.
    source = write_model_dir(tmp_path / "release", ood_threshold=20.0)
    dest = tmp_path / "models"
    download_model(source.as_uri(), dest)
    assert (dest / "ood_stats.npz").read_bytes() == (source / "ood_stats.npz").read_bytes()
    assert load_artifacts(dest).ood_mean is not None


@pytest.mark.parametrize("name", ["../escape.bin", "sub/file.bin", ".hidden", "model_meta.json"])
def test_unsafe_file_names_are_refused(tmp_path: Path, name: str) -> None:
    source = write_model_dir(tmp_path / "release")
    meta = json.loads((source / "model_meta.json").read_text(encoding="utf-8"))
    meta["files"][name] = {"sha256": "0" * 64}
    (source / "model_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ModelArtifactError, match="Unsafe"):
        download_model(source.as_uri(), tmp_path / "models")
    assert not (tmp_path / "escape.bin").exists()
