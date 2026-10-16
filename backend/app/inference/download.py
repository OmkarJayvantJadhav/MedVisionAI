"""Download model artifacts and verify them against their recorded checksums.

``python -m app.inference.download --base-url URL [--dest models]``

``URL`` is a folder (for example a GitHub release download URL) that serves
``model_meta.json``, ``model.onnx`` and ``head_weights.npz``. Files are written
atomically and only after their SHA-256 matches the metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from collections.abc import Sequence
from pathlib import Path

from app.inference.artifacts import META_FILE, MODEL_FILE, WEIGHTS_FILE, ModelMeta, sha256_file
from app.inference.errors import ModelArtifactError

CHUNK = 1 << 20
DEFAULT_DEST = Path(__file__).resolve().parents[3] / "models"


def _url(base_url: str, name: str) -> str:
    return f"{base_url.rstrip('/')}/{name}"


def _fetch(url: str, timeout: float) -> bytes:
    if not url.startswith(("https://", "http://", "file://")):
        raise ModelArtifactError(f"Unsupported URL scheme: {url}")
    with urllib.request.urlopen(url, timeout=timeout) as response:
        data: bytes = response.read()
    return data


def _download_verified(url: str, destination: Path, expected_sha256: str, timeout: float) -> None:
    """Stream ``url`` to a temporary file, verify it, then move it into place."""
    if not url.startswith(("https://", "http://", "file://")):
        raise ModelArtifactError(f"Unsupported URL scheme: {url}")
    digest = hashlib.sha256()
    fd, tmp_name = tempfile.mkstemp(dir=destination.parent, prefix=f".{destination.name}.")
    try:
        with (
            os.fdopen(fd, "wb") as handle,
            urllib.request.urlopen(url, timeout=timeout) as response,
        ):
            for chunk in iter(lambda: response.read(CHUNK), b""):
                digest.update(chunk)
                handle.write(chunk)
        if digest.hexdigest() != expected_sha256:
            raise ModelArtifactError(f"Checksum mismatch for {url}")
        Path(tmp_name).replace(destination)
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def download_model(base_url: str, dest: Path, *, timeout: float = 60.0) -> ModelMeta:
    """Download and verify the artifacts into ``dest``; skip files already present and valid.

    Raises:
        ModelArtifactError: On invalid metadata or a checksum mismatch.
        OSError: On network or filesystem errors.
    """
    dest.mkdir(parents=True, exist_ok=True)
    meta_bytes = _fetch(_url(base_url, META_FILE), timeout)
    try:
        meta = ModelMeta.from_dict(json.loads(meta_bytes))
    except json.JSONDecodeError as error:
        raise ModelArtifactError("Downloaded metadata is not valid JSON.") from error

    for name in (MODEL_FILE, WEIGHTS_FILE):
        expected = meta.checksums.get(name)
        if expected is None:
            raise ModelArtifactError(f"No checksum recorded for {name}")
        target = dest / name
        if target.is_file() and sha256_file(target) == expected:
            continue
        _download_verified(_url(base_url, name), target, expected, timeout)
    # Write the metadata last so a partial download never looks complete.
    (dest / META_FILE).write_bytes(meta_bytes)
    return meta


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-url", required=True, help="folder URL serving the artifacts")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)
    try:
        meta = download_model(args.base_url, args.dest, timeout=args.timeout)
    except (ModelArtifactError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"Model {meta.model_version} ready in {args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
