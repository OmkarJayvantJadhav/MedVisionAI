"""Training, evaluation, calibration and export pipeline for MedVision AI."""

import os

# This package only writes figures to files. Notebook hosts such as Colab set
# MPLBACKEND to an inline backend that is not installed in this environment,
# which makes Matplotlib fail on import, so always use the headless backend.
os.environ["MPLBACKEND"] = "agg"

__version__ = "0.1.0"
