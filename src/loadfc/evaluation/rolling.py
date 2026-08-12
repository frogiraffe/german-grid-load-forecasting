"""One-step rolling forecast engine, shared by all forecasters."""

from __future__ import annotations

import pandas as pd

from ..models.base import BaseForecaster


def rolling_forecast(
    model: BaseForecaster,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    exog_cols: list[str],
) -> pd.Series:
    model.fit(train_df, exog_cols)
    preds = []
    for _, row in test_df.iterrows():
        exog_row = row[exog_cols]
        preds.append(float(model.predict_next(exog_row)))
        model.update(float(row["daily_load"]), exog_row)
    return pd.Series(preds, index=test_df.index, name=model.name)
