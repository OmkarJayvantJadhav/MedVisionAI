"""Static checks on the Kaggle training notebook, which CI cannot run."""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks" / "train_kaggle.ipynb"


def _code() -> str:
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    return "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")


def test_headless_matplotlib_is_set_before_tensorflow_is_imported() -> None:
    # Regression: Kaggle sets MPLBACKEND to its inline backend, which the
    # project environment lacks; the GPU check imports TensorFlow (and with it
    # Matplotlib) before medvision_ml can override it, so the run failed.
    code = _code()
    setting = code.find('os.environ["MPLBACKEND"] = "agg"')
    first_import = code.find("import tensorflow")
    assert setting != -1
    assert first_import != -1
    assert setting < first_import
