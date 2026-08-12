"""Prequential telemetry for rolling backtest predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_backtest_telemetry(
    predictions: pd.DataFrame,
    *,
    window: int,
    mape_warning: float,
    coverage_floor: float,
    lower_column: str | None = None,
    upper_column: str | None = None,
) -> pd.DataFrame:
    """Compute outcome-time error, coverage, and rolling alert state."""

    if window < 2:
        raise ValueError("telemetry window must be at least 2")
    if mape_warning <= 0:
        raise ValueError("MAPE warning threshold must be positive")
    if not 0 < coverage_floor < 1:
        raise ValueError("coverage floor must be between 0 and 1")
    required = {"actual", "forecast"}
    if not required <= set(predictions):
        raise ValueError("telemetry predictions require actual and forecast columns")
    if (predictions["actual"] == 0).any():
        raise ValueError("telemetry MAPE is undefined for zero actuals")
    if (lower_column is None) != (upper_column is None):
        raise ValueError("telemetry intervals require both lower and upper columns")
    if lower_column is not None and not {lower_column, upper_column} <= set(predictions):
        raise ValueError("telemetry interval columns are missing")

    out = predictions[["actual", "forecast"]].copy()
    out["error"] = out["actual"] - out["forecast"]
    out["absolute_percentage_error"] = np.abs(out["error"] / out["actual"]) * 100
    out["rolling_mape"] = (
        out["absolute_percentage_error"].rolling(window, min_periods=window).mean()
    )
    out["rolling_bias"] = out["error"].rolling(window, min_periods=window).mean()
    out["mape_alert"] = out["rolling_mape"] > mape_warning

    if lower_column is None or upper_column is None:
        out["covered"] = pd.NA
        out["interval_width"] = np.nan
        out["rolling_coverage"] = np.nan
        out["coverage_alert"] = False
    else:
        out["covered"] = (predictions["actual"] >= predictions[lower_column]) & (
            predictions["actual"] <= predictions[upper_column]
        )
        out["interval_width"] = predictions[upper_column] - predictions[lower_column]
        out["rolling_coverage"] = (
            out["covered"].astype(float).rolling(window, min_periods=window).mean()
        )
        out["coverage_alert"] = out["rolling_coverage"] < coverage_floor
    return out
