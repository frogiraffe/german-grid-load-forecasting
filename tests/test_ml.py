from datetime import date, timedelta

import numpy as np
import pandas as pd

from loadfc.evaluation.rolling import rolling_forecast
from loadfc.models.ml import make_ml_forecaster


def _frame(n, seed=1):
    rng = np.random.default_rng(seed)
    idx = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 500 + 3 * x1 - 2 * x2 + rng.normal(scale=1, size=n)
    return pd.DataFrame({"daily_load": y, "x1": x1, "x2": x2}, index=idx)


def test_each_ml_model_runs_and_predicts():
    df = _frame(120)
    train, test = df.iloc[:90], df.iloc[90:]
    for kind in ["xgboost", "lightgbm", "random_forest"]:
        model = make_ml_forecaster(kind, {"n_estimators": 20, "random_state": 0})
        preds = rolling_forecast(model, train, test, ["x1", "x2"])
        assert len(preds) == len(test)
        assert preds.notna().all()
        assert model.name == kind


def test_unknown_model_raises():
    import pytest

    with pytest.raises(ValueError, match="unknown"):
        make_ml_forecaster("svm", {})
