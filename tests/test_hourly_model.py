import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from loadfc.config import Config
from loadfc.models.hourly import (
    HourlyDirectForecaster,
    HourlyHybridForecaster,
    make_hourly_direct_lightgbm,
    make_hourly_direct_ridge,
    make_hourly_hybrid,
    make_hourly_quantile_lightgbm,
)
from loadfc.models.hybrid import HybridResidualRegressor


def _training_frame() -> pd.DataFrame:
    origin = pd.date_range("2024-01-01", periods=12, freq="h", tz="UTC")
    horizon = np.tile([1, 2, 3], 4)
    return pd.DataFrame(
        {
            "hourly_load": 100 + np.arange(12) * 2 + horizon,
            "trend_hour": np.arange(12),
            "horizon": horizon,
            "forecast_origin": origin,
            "valid_time": origin + pd.to_timedelta(horizon, unit="h"),
        }
    )


def test_hourly_hybrid_fits_and_exposes_prediction_components():
    estimator = HybridResidualRegressor(
        LinearRegression(), DecisionTreeRegressor(max_depth=2, random_state=42)
    )
    model = HourlyHybridForecaster(estimator).fit(_training_frame())

    prediction = model.predict(_training_frame().iloc[-3:])

    assert list(prediction) == ["baseline", "residual_correction", "prediction"]
    assert np.allclose(
        prediction["prediction"],
        prediction["baseline"] + prediction["residual_correction"],
    )
    assert model.feature_columns == ("trend_hour", "horizon")


def test_hourly_hybrid_rejects_missing_horizon_feature():
    with pytest.raises(ValueError, match="include horizon"):
        HourlyHybridForecaster(
            HybridResidualRegressor(LinearRegression(), DecisionTreeRegressor())
        ).fit(_training_frame(), feature_columns=["trend_hour"])


def test_configured_hourly_hybrid_uses_scaled_ridge_baseline():
    model = make_hourly_hybrid(Config.from_yaml("config.yaml"))

    assert list(model.estimator.baseline.named_steps) == ["standardscaler", "ridge"]


@pytest.mark.parametrize(
    "model",
    [
        make_hourly_direct_ridge(),
        make_hourly_direct_lightgbm(Config.from_yaml("config.yaml")),
    ],
)
def test_hourly_direct_ablation_fits_and_predicts(model):
    prediction = model.fit(_training_frame()).predict(_training_frame().iloc[-3:])

    assert list(prediction) == ["prediction"]
    assert len(prediction) == 3
    assert np.isfinite(prediction["prediction"]).all()


def test_hourly_direct_rejects_prediction_before_fit():
    with pytest.raises(ValueError, match="must be fitted"):
        HourlyDirectForecaster(LinearRegression()).predict(_training_frame())


def test_hourly_quantile_lightgbm_configures_quantile_objective():
    model = make_hourly_quantile_lightgbm(
        Config.from_yaml("config.yaml"),
        quantile=0.05,
    )

    assert model.estimator.objective == "quantile"
    assert model.estimator.alpha == pytest.approx(0.05)


def test_hourly_quantile_lightgbm_rejects_invalid_quantile():
    with pytest.raises(ValueError, match="between 0 and 1"):
        make_hourly_quantile_lightgbm(Config.from_yaml("config.yaml"), quantile=1.0)
