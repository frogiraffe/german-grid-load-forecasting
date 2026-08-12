"""Optuna hyperparameter search for the tree-based load forecasters.

The committed values in ``config.yaml`` are the frozen baseline behind every
number in the README. This package is the code that shows how they were found:
``scripts/run_tune.py`` runs a study and writes its result to ``results/tuning/``
without touching ``config.yaml``.
"""

from .search_spaces import SPACES, suggest_params
from .study import cv_mase, training_slice, tune_model

__all__ = ["SPACES", "suggest_params", "cv_mase", "training_slice", "tune_model"]
