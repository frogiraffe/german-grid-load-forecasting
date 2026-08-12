from datetime import date, timedelta

import pandas as pd

from loadfc.evaluation.rolling import rolling_forecast
from loadfc.models.base import BaseForecaster


class _LastValueForecaster(BaseForecaster):
    name = "last"

    def fit(self, train_df, exog_cols):
        self._last = float(train_df["daily_load"].iloc[-1])

    def predict_next(self, exog_row):
        return self._last

    def update(self, y_actual, exog_row):
        self._last = float(y_actual)


def _frame(start, vals):
    idx = [start + timedelta(days=i) for i in range(len(vals))]
    return pd.DataFrame({"daily_load": vals, "x": [0.0] * len(vals)}, index=idx)


def test_rolling_forecast_uses_only_past_information():
    train = _frame(date(2024, 1, 1), [10.0, 20.0, 30.0])
    test = _frame(date(2024, 1, 4), [40.0, 50.0])
    preds = rolling_forecast(_LastValueForecaster(), train, test, ["x"])
    assert list(preds) == [30.0, 40.0]
    assert list(preds.index) == list(test.index)
