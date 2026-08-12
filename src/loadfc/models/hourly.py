"""Fit and predict a global direct multi-horizon residual-hybrid model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..config import Config
from .hybrid import HybridResidualRegressor

_NON_FEATURES = {"hourly_load", "forecast_origin", "valid_time"}


class HourlyHybridForecaster:
    """One global model whose horizon column distinguishes forecast steps."""

    def __init__(self, estimator: HybridResidualRegressor) -> None:
        self.estimator = estimator
        self.feature_columns: tuple[str, ...] | None = None

    def fit(
        self,
        training_frame: pd.DataFrame,
        feature_columns: Sequence[str] | None = None,
    ) -> HourlyHybridForecaster:
        columns = (
            list(feature_columns)
            if feature_columns is not None
            else [
                column
                for column in training_frame.select_dtypes("number").columns
                if column not in _NON_FEATURES
            ]
        )
        if "horizon" not in columns:
            raise ValueError("hourly hybrid features must include horizon")
        if not columns:
            raise ValueError("hourly hybrid requires at least one predictor")
        clean = training_frame.dropna(subset=["hourly_load", *columns])
        if clean.empty:
            raise ValueError("hourly hybrid has no complete training rows")
        self.feature_columns = tuple(columns)
        self.estimator.fit(clean[columns], clean["hourly_load"])
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.feature_columns is None:
            raise ValueError("hourly hybrid must be fitted before prediction")
        missing = set(self.feature_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"hourly prediction is missing features: {sorted(missing)}")
        if frame[list(self.feature_columns)].isna().any().any():
            raise ValueError("hourly prediction features contain missing values")
        baseline, residual = self.estimator.predict_components(frame[list(self.feature_columns)])
        return pd.DataFrame(
            {
                "baseline": baseline,
                "residual_correction": residual,
                "prediction": baseline + residual,
            },
            index=frame.index,
        )


class HourlyDirectForecaster:
    """Direct multi-horizon forecaster trained on the load level."""

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator
        self.estimator_: Any | None = None
        self.feature_columns: tuple[str, ...] | None = None

    def fit(
        self,
        training_frame: pd.DataFrame,
        feature_columns: Sequence[str] | None = None,
    ) -> HourlyDirectForecaster:
        columns = (
            list(feature_columns)
            if feature_columns is not None
            else [
                column
                for column in training_frame.select_dtypes("number").columns
                if column not in _NON_FEATURES
            ]
        )
        if "horizon" not in columns:
            raise ValueError("hourly direct features must include horizon")
        clean = training_frame.dropna(subset=["hourly_load", *columns])
        if clean.empty:
            raise ValueError("hourly direct model has no complete training rows")
        self.feature_columns = tuple(columns)
        self.estimator_ = clone(self.estimator).fit(clean[columns], clean["hourly_load"])
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.feature_columns is None or self.estimator_ is None:
            raise ValueError("hourly direct model must be fitted before prediction")
        missing = set(self.feature_columns) - set(frame.columns)
        if missing:
            raise ValueError(f"hourly prediction is missing features: {sorted(missing)}")
        if frame[list(self.feature_columns)].isna().any().any():
            raise ValueError("hourly prediction features contain missing values")
        prediction = self.estimator_.predict(frame[list(self.feature_columns)])
        return pd.DataFrame({"prediction": prediction}, index=frame.index)


def make_hourly_hybrid(cfg: Config) -> HourlyHybridForecaster:
    """Build the configured linear-trend plus LightGBM residual model."""

    params = dict(cfg.models["lightgbm"])
    params.setdefault("random_state", cfg.seed)
    params.setdefault("n_jobs", 1)
    residual = LGBMRegressor(**{"verbosity": -1, **params})
    baseline = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    return HourlyHybridForecaster(HybridResidualRegressor(baseline, residual))


def make_hourly_direct_ridge() -> HourlyDirectForecaster:
    """Build a scaled Ridge load-level ablation."""

    return HourlyDirectForecaster(make_pipeline(StandardScaler(), Ridge(alpha=1.0)))


def make_hourly_direct_lightgbm(cfg: Config) -> HourlyDirectForecaster:
    """Build a LightGBM load-level ablation with the configured parameters."""

    params = dict(cfg.models["lightgbm"])
    params.setdefault("random_state", cfg.seed)
    params.setdefault("n_jobs", 1)
    return HourlyDirectForecaster(LGBMRegressor(**{"verbosity": -1, **params}))


def make_hourly_quantile_lightgbm(
    cfg: Config,
    *,
    quantile: float,
) -> HourlyDirectForecaster:
    """Build a LightGBM quantile head for CQR."""

    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1")
    params = dict(cfg.models["lightgbm"])
    params.setdefault("random_state", cfg.seed)
    params.setdefault("n_jobs", 1)
    params.update({"objective": "quantile", "alpha": quantile})
    return HourlyDirectForecaster(LGBMRegressor(**{"verbosity": -1, **params}))
