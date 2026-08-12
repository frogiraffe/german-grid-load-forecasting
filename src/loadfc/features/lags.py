"""Lagged target features (machine-learning models only)."""

from __future__ import annotations

import pandas as pd


def add_lags(target: pd.Series, lags: list[int]) -> pd.DataFrame:
    return pd.DataFrame({f"L_t-{k}": target.shift(k) for k in lags}, index=target.index)
