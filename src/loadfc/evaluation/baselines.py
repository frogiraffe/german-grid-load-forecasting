"""Explicit non-trained reference forecasts."""

from __future__ import annotations

import pandas as pd

BASELINE_COLUMNS = {"naive_1d": "L_t-1", "seasonal_naive_7d": "L_t-7"}


def baseline_predictions(features: pd.DataFrame, start, end) -> dict[str, pd.Series]:
    evaluation = features[(features.index >= start) & (features.index <= end)]
    return {
        name: evaluation[column].dropna().rename(name) for name, column in BASELINE_COLUMNS.items()
    }
