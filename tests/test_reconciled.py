from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from loadfc.models.reconciled import ReconciledForecaster


def _model_and_input() -> tuple[ReconciledForecaster, pd.DataFrame]:
    hourly_training = pd.DataFrame({"hourly_feature": np.arange(24, dtype="float64")})
    hourly = LinearRegression().fit(
        hourly_training,
        100.0 + hourly_training["hourly_feature"].to_numpy(),
    )
    daily = DummyRegressor(strategy="constant", constant=240.0).fit(
        np.array([[0.0], [1.0]]),
        np.array([240.0, 240.0]),
    )
    model = ReconciledForecaster(
        hourly_estimator=hourly,
        daily_estimator=daily,
        hourly_feature_columns=("hourly_feature",),
        daily_feature_columns=("daily_feature",),
    )
    frame = hourly_training.copy()
    frame["reconciliation_date"] = "2026-07-01"
    frame["daily__daily_feature"] = 1.0
    return model, frame


def test_reconciled_forecaster_matches_daily_anchor():
    model, frame = _model_and_input()

    prediction = model.predict(frame)

    assert prediction.mean() == pytest.approx(240.0)
    assert np.isfinite(prediction).all()


def test_reconciled_forecaster_rejects_incomplete_local_day():
    model, frame = _model_and_input()

    with pytest.raises(ValueError, match="complete local days"):
        model.predict(frame.iloc[:-1])
