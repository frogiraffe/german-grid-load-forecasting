from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from loadfc.config import Config
from loadfc.features.hourly import build_hourly_features, hourly_feature_matrix


def _frame(periods: int = 200) -> pd.DataFrame:
    index = pd.date_range("2023-12-24", periods=periods, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "hourly_load": np.arange(periods, dtype="float64") + 50_000,
            "Temp_t": np.arange(periods, dtype="float64"),
            "Wind_t": np.arange(periods, dtype="float64") / 10,
            "Temp_operational": np.nan,
            "Wind_operational": np.nan,
        },
        index=index,
    )


def test_hourly_features_use_24_hour_persistence_without_future_leakage():
    cfg = Config.from_yaml("config.yaml")
    frame = _frame()
    valid_time = frame.index[48]

    features = build_hourly_features(frame, cfg)
    frame.loc[valid_time, "Temp_t"] = 9999.0
    changed = build_hourly_features(frame, cfg)

    assert features.loc[valid_time, "Temp_forecast"] == frame["Temp_t"].iloc[24]
    assert changed.loc[valid_time, "Temp_forecast"] == features.loc[valid_time, "Temp_forecast"]
    assert features.loc[valid_time, "L_t-24"] == frame["hourly_load"].iloc[24]


def test_operational_weather_wins_when_available():
    cfg = Config.from_yaml("config.yaml")
    frame = _frame()
    valid_time = frame.index[48]
    frame.loc[valid_time, ["Temp_operational", "Wind_operational"]] = [7.0, 9.0]

    features = build_hourly_features(frame, cfg)

    assert features.loc[valid_time, "Temp_forecast"] == 7.0
    assert features.loc[valid_time, "Wind_forecast"] == 9.0


@pytest.mark.filterwarnings("error::FutureWarning")
def test_available_day_ahead_fallback_is_warning_free_and_dtype_stable():
    cfg = Config.from_yaml("config.yaml")
    cfg = replace(cfg, features={**cfg.features, "weather_strategy": "available_day_ahead"})
    frame = _frame()
    frame["Temp_t"] = np.arange(len(frame), dtype="int64")
    frame["Temp_operational"] = np.arange(1000, 1000 + len(frame), dtype="int64")
    frame["Wind_operational"] = np.arange(2000, 2000 + len(frame), dtype="float64")
    fallback_time = frame.index[48]
    frame.loc[fallback_time, "Wind_operational"] = np.nan

    features = build_hourly_features(frame, cfg)

    assert features["Temp_forecast"].tolist() == list(range(1000, 1000 + len(frame)))
    assert str(features["Temp_forecast"].dtype) == "int64"
    assert features.loc[fallback_time, "Wind_forecast"] == pytest.approx(2.4)
    assert features.loc[frame.index[47], "Wind_forecast"] == 2047.0
    assert str(features["Wind_forecast"].dtype) == "float64"


def test_hourly_matrix_drops_only_unavailable_history_rows():
    cfg = Config.from_yaml("config.yaml")
    matrix = hourly_feature_matrix(build_hourly_features(_frame(), cfg))

    assert matrix.index.min() == _frame().index[168]
    assert {"hour", "sin_hour", "cos_hour", "L_t-24", "L_t-168"} <= set(matrix)


def test_fourier_phase_is_stable_across_dataframe_slices():
    cfg = Config.from_yaml("config.yaml")
    frame = _frame()
    complete = build_hourly_features(frame, cfg)
    sliced = build_hourly_features(frame.iloc[24:], cfg)
    timestamp = frame.index[100]

    for column in ("sin_1_week", "cos_1_week", "sin_1_year", "cos_1_year"):
        assert sliced.loc[timestamp, column] == complete.loc[timestamp, column]


def test_hourly_time_basis_is_independent_of_datetime_storage_resolution():
    cfg = Config.from_yaml("config.yaml")
    frame = _frame().copy()
    frame.index = frame.index.as_unit("us")

    features = build_hourly_features(frame, cfg)

    assert features["trend_hour"].diff().dropna().eq(1.0).all()
    assert features["sin_1_week"].iloc[168] == pytest.approx(features["sin_1_week"].iloc[0])
    assert features["cos_1_week"].iloc[168] == pytest.approx(features["cos_1_week"].iloc[0])


def test_hourly_climatology_freezes_at_berlin_validation_midnight():
    cfg = Config.from_yaml("config.yaml")
    cfg = replace(cfg, split=replace(cfg.split, train_end=date(2023, 12, 31)))
    frame = _frame(96)
    frame.index = pd.date_range("2023-12-30", periods=len(frame), freq="h", tz="UTC")
    cutoff = pd.Timestamp("2024-01-01", tz="Europe/Berlin").tz_convert("UTC")

    features = build_hourly_features(frame, cfg)
    frame.loc[cutoff, "Temp_t"] = 9999.0
    changed = build_hourly_features(frame, cfg)

    assert cutoff == pd.Timestamp("2023-12-31T23:00Z")
    assert changed.loc[cutoff, "Temp_anomaly"] == features.loc[cutoff, "Temp_anomaly"]
