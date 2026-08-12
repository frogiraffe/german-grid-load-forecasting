"""Rolling-origin evaluation over fixed six-month test windows."""

from __future__ import annotations

from datetime import date

import pandas as pd

from .metrics import mape
from .rolling import rolling_forecast


def six_month_windows(start: date, end: date) -> list[tuple[str, date, date]]:
    windows: list[tuple[str, date, date]] = []
    year = start.year
    while year <= end.year:
        halves = [
            (f"{year}-H1", date(year, 1, 1), date(year, 6, 30)),
            (f"{year}-H2", date(year, 7, 1), date(year, 12, 31)),
        ]
        for label, lo, hi in halves:
            if hi >= start and lo <= end:
                windows.append((label, max(lo, start), min(hi, end)))
        year += 1
    return windows


def rolling_origin_mape(
    model_factory,
    full_df: pd.DataFrame,
    exog_cols: list[str],
    windows: list[tuple[str, date, date]],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for label, lo, hi in windows:
        train = full_df[full_df.index < lo]
        test = full_df[(full_df.index >= lo) & (full_df.index <= hi)]
        preds = rolling_forecast(model_factory(), train, test, exog_cols)
        out[label] = mape(test["daily_load"], preds)
    return out
