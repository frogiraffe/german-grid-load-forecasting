from datetime import date, timedelta

import numpy as np
import pandas as pd

from loadfc.evaluation.rolling import rolling_forecast
from loadfc.models.sarimax import SarimaxForecaster


def _frame(n, seed=0):
    rng = np.random.default_rng(seed)
    idx = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    week = np.sin(2 * np.pi * np.arange(n) / 7)
    x = rng.normal(size=n)
    y = 1000 + 50 * week + 10 * x + rng.normal(scale=5, size=n)
    return pd.DataFrame({"daily_load": y, "x": x}, index=idx)


def test_sarimax_rolling_produces_one_pred_per_test_day():
    df = _frame(80)
    train, test = df.iloc[:60], df.iloc[60:]
    model = SarimaxForecaster(order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), refit="false")
    preds = rolling_forecast(model, train, test, ["x"])
    assert len(preds) == len(test)
    assert preds.notna().all()
    assert preds.between(800, 1200).all()


def test_sarimax_refit_modes_run():
    df = _frame(60)
    train, test = df.iloc[:48], df.iloc[48:]
    for refit in ["false", "true", "periodic"]:
        model = SarimaxForecaster((1, 0, 0), (0, 0, 0, 0), refit=refit, refit_period=3)
        preds = rolling_forecast(model, train, test, ["x"])
        assert len(preds) == len(test)
