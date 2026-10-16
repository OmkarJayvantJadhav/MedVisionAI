"""Regression test: plotting must work when the host sets a notebook-only backend."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_plots_work_under_colab_backend_setting(tmp_path: Path) -> None:
    env = {**os.environ, "MPLBACKEND": "module://matplotlib_inline.backend_inline"}
    code = (
        "from pathlib import Path\n"
        "from medvision_ml.evaluation.plots import plot_training_history\n"
        f"plot_training_history({{'loss': [1.0, 0.5]}}, Path({str(tmp_path / 'h.png')!r}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "h.png").is_file()
