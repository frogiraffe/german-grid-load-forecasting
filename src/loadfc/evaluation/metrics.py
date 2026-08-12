"""Forecast error metrics: RMSE, MAE, MAPE, MASE."""

from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    y = np.asarray(y_true, dtype="float64")
    p = np.asarray(y_pred, dtype="float64")
    return float(100.0 * np.mean(np.abs((y - p) / y)))


def seasonal_naive_mae(train: pd.Series, season: int) -> float:
    s = np.asarray(train, dtype="float64")
    return float(np.mean(np.abs(s[season:] - s[:-season])))


def mase(y_true: pd.Series, y_pred: pd.Series, naive_mae: float) -> float:
    return mae(y_true, y_pred) / naive_mae


def all_metrics(y_true: pd.Series, y_pred: pd.Series, naive_mae: float) -> dict[str, float]:
    return {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "MASE": mase(y_true, y_pred, naive_mae),
    }
