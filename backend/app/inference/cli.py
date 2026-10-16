"""Run the inference core on image files: ``python -m app.inference.cli IMAGE...``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from app.inference.errors import InferenceError
from app.inference.predictor import Predictor

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[3] / "models"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--overlay-dir", type=Path, default=None, help="save Grad-CAM overlays here"
    )
    args = parser.parse_args(argv)

    try:
        predictor = Predictor.from_directory(args.model_dir)
    except InferenceError as error:
        print(f"error: {error.message}", file=sys.stderr)
        return 2

    exit_code = 0
    for path in args.images:
        try:
            prediction = predictor.predict(path.read_bytes(), explain=args.overlay_dir is not None)
        except (OSError, InferenceError) as error:
            print(f"{path}: error: {error}", file=sys.stderr)
            exit_code = 1
            continue
        result = asdict(prediction)
        explanation = result.pop("explanation")
        if explanation is not None and args.overlay_dir is not None:
            args.overlay_dir.mkdir(parents=True, exist_ok=True)
            overlay_path = args.overlay_dir / f"{path.stem}_overlay.png"
            overlay_path.write_bytes(explanation["overlay_png"])
            result["overlay"] = str(overlay_path)
        print(json.dumps({"image": str(path), **result}, indent=2, default=str))
    print("Not a medical diagnosis. Consult a qualified dermatologist.", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
