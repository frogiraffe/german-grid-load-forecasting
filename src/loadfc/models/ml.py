"""Tree-based ML forecasters with a single fit and lag-fed predictions."""

from __future__ import annotations

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from .base import BaseForecaster


class MLForecaster(BaseForecaster):
    def __init__(self, name: str, estimator):
        self.name = name
        self._estimator = estimator

    def fit(self, train_df: pd.DataFrame, exog_cols: list[str]) -> None:
        self._estimator.fit(
            train_df[exog_cols].to_numpy(dtype="float64"),
            train_df["daily_load"].to_numpy(dtype="float64"),
        )

    def predict_next(self, exog_row: pd.Series) -> float:
        x = exog_row.to_numpy(dtype="float64").reshape(1, -1)
        return float(self._estimator.predict(x)[0])


def make_estimator(kind: str, params: dict):
    """Build the raw regressor for `kind`, shared by the rolling forecaster and
    the cross-validated tuner (which fits/predicts on whole folds)."""
    params = {"n_jobs": 1, **params}
    if kind == "xgboost":
        return XGBRegressor(**params)
    if kind == "lightgbm":
        return LGBMRegressor(**{"verbosity": -1, **params})
    if kind == "random_forest":
        return RandomForestRegressor(**params)
    raise ValueError(f"unknown ML model: {kind!r}")


def make_ml_forecaster(kind: str, params: dict) -> MLForecaster:
    return MLForecaster(kind, make_estimator(kind, params))
