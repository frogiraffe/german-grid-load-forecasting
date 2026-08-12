"""Cross-validated Optuna study over the tree-based forecasters.

Tuning must never see the validation or test period, so callers slice the
feature matrix to the training partition and score each trial with an
expanding-window ``TimeSeriesSplit``. The objective is mean MASE — the same
seasonal-naive-scaled error the study minimises. Because the ML lag features
hold *actual* past loads, a batch fit-and-predict on a validation fold is
identical to the one-step rolling forecast used at test time, so the study
optimises exactly what the final evaluation reports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from ..evaluation.metrics import mae, seasonal_naive_mae
from ..models.ml import make_estimator
from .search_spaces import suggest_params


@dataclass(frozen=True)
class TuneResult:
    kind: str
    params: dict
    value: float
    n_trials: int


def training_slice(matrix: pd.DataFrame, test_start) -> pd.DataFrame:
    """Rows strictly before `test_start` — the only data tuning is allowed to see.

    The processed matrix is indexed by ``datetime.date`` objects, so normalise to
    a DatetimeIndex before comparing rather than assume either type.
    """
    return matrix[pd.DatetimeIndex(matrix.index) < pd.Timestamp(test_start)]


def cv_mase(kind, matrix, exog_cols, params, n_splits, seed, season: int = 7) -> float:
    """Mean MASE over an expanding-window time-series cross-validation.

    Each fold's MASE scales the model's validation MAE by the seasonal-naive MAE
    of that fold's training slice, so the score is comparable across folds of
    different length.
    """
    X = matrix[exog_cols].to_numpy(dtype="float64")
    y = matrix["daily_load"].to_numpy(dtype="float64")
    scores: list[float] = []
    for train_idx, val_idx in TimeSeriesSplit(n_splits=n_splits).split(X):
        est = make_estimator(kind, {**params, "random_state": seed})
        est.fit(X[train_idx], y[train_idx])
        naive = seasonal_naive_mae(y[train_idx], season=season)
        model_mae = mae(y[val_idx], est.predict(X[val_idx]))
        scores.append(model_mae / naive)
    return float(np.mean(scores))


def tune_model(kind, matrix, exog_cols, n_trials, n_splits, seed) -> TuneResult:
    """Run a seeded TPE study for `kind` and return the best hyperparameters."""

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(kind, trial)
        return cv_mase(kind, matrix, exog_cols, params, n_splits, seed)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return TuneResult(
        kind=kind, params=study.best_params, value=study.best_value, n_trials=n_trials
    )
