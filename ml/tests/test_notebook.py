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


def test_every_experiment_is_a_valid_configuration() -> None:
    # A typo in an override would otherwise only surface hours into a Kaggle run.
    from medvision_ml.config import DEFAULT_CONFIG_PATH, load_config

    code = _code()
    start = code.index('EXPERIMENT = "')
    end = code.index("for key, value in EXPERIMENTS[EXPERIMENT].items():")
    namespace: dict[str, object] = {}
    exec(code[start:end], namespace)  # the notebook's own literal definitions
    experiments = namespace["EXPERIMENTS"]
    assert isinstance(experiments, dict)
    assert namespace["EXPERIMENT"] in experiments
    assert experiments["baseline"] == {}
    for name, overrides in experiments.items():
        environ = {f"MEDVISION__{key}": str(value) for key, value in overrides.items()}
        config = load_config(DEFAULT_CONFIG_PATH, environ=environ)
        assert config is not None, name
