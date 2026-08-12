"""SARIMAX forecaster with a configurable rolling-update strategy."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .base import BaseForecaster


class SarimaxForecaster(BaseForecaster):
    name = "SARIMAX"

    def __init__(self, order, seasonal_order, refit="false", refit_period=90):
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.refit = refit
        self.refit_period = refit_period
        self._res = None
        self._steps_since_refit = 0

    def fit(self, train_df: pd.DataFrame, exog_cols: list[str]) -> None:
        endog = train_df["daily_load"].to_numpy(dtype="float64")
        exog = train_df[exog_cols].to_numpy(dtype="float64")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                endog,
                exog=exog,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self._res = model.fit(disp=False)

    def predict_next(self, exog_row: pd.Series) -> float:
        exog = exog_row.to_numpy(dtype="float64").reshape(1, -1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = self._res.get_forecast(steps=1, exog=exog)
        return float(fc.predicted_mean[0])

    def update(self, y_actual: float, exog_row: pd.Series) -> None:
        endog = np.array([y_actual], dtype="float64")
        exog = exog_row.to_numpy(dtype="float64").reshape(1, -1)
        self._steps_since_refit += 1
        do_refit = self.refit == "true" or (
            self.refit == "periodic" and self._steps_since_refit >= self.refit_period
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._res = self._res.append(endog=endog, exog=exog, refit=do_refit)
        if do_refit:
            self._steps_since_refit = 0
