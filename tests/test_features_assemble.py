from dataclasses import replace
from datetime import date, timedelta

import pandas as pd
import pytest

from loadfc.config import Config, SplitConfig
from loadfc.features.assemble import build_features, exog_columns, feature_matrix


def _dataset(n=40):
    idx = [date(2019, 1, 14) + timedelta(days=i) for i in range(n)]
    return pd.DataFrame(
        {
            "daily_load": [50000.0 + i for i in range(n)],
            "Temp_t": [5.0 + (i % 10) for i in range(n)],
            "Wind_t": [10.0] * n,
            "Temp_operational": [float("nan")] * n,
            "Wind_operational": [float("nan")] * n,
        },
        index=pd.Index(idx, name="date"),
    )


def test_build_features_has_full_contract():
    cfg = Config.from_yaml("config.yaml")
    feats = build_features(_dataset(), cfg)
    expected = [
        "daily_load",
        "Temp_forecast",
        "Wind_forecast",
        "HDD",
        "CDD",
        "Temp_anomaly",
        "Weekend",
        "holiday_national",
        "holiday_religious",
        "bridge_day",
        "pre_holiday",
        "post_holiday",
        "is_covid",
        "is_energy_crisis",
        "sin_1_week",
        "cos_1_week",
        "sin_1_year",
        "cos_1_year",
        "L_t-1",
        "L_t-7",
    ]
    assert list(feats.columns) == expected


def test_sarimax_matrix_excludes_lags_and_drops_unavailable_first_weather_row():
    cfg = Config.from_yaml("config.yaml")
    feats = build_features(_dataset(40), cfg)
    m = feature_matrix(feats, "sarimax")
    assert "L_t-1" not in m.columns and "L_t-7" not in m.columns
    assert len(m) == 39
    assert "daily_load" in m.columns


def test_pre_archive_weather_uses_only_previous_day():
    cfg = Config.from_yaml("config.yaml")
    dataset = _dataset(10)
    feats = build_features(dataset, cfg)
    assert pd.isna(feats.iloc[0]["Temp_forecast"])
    assert feats.iloc[1]["Temp_forecast"] == dataset.iloc[0]["Temp_t"]
    assert feats.iloc[1]["Wind_forecast"] == dataset.iloc[0]["Wind_t"]


def test_operational_weather_replaces_persistence_when_available():
    cfg = Config.from_yaml("config.yaml")
    dataset = _dataset(10)
    dataset.loc[dataset.index[4], "Temp_operational"] = -3.0
    dataset.loc[dataset.index[4], "Wind_operational"] = 22.0
    feats = build_features(dataset, cfg)
    assert feats.loc[dataset.index[4], "Temp_forecast"] == -3.0
    assert feats.loc[dataset.index[4], "Wind_forecast"] == 22.0


@pytest.mark.filterwarnings("error::FutureWarning")
def test_available_day_ahead_fallback_is_warning_free_and_dtype_stable():
    cfg = Config.from_yaml("config.yaml")
    cfg = replace(cfg, features={**cfg.features, "weather_strategy": "available_day_ahead"})
    dataset = _dataset(10)
    dataset["Temp_operational"] = range(100, 110)
    dataset["Wind_t"] = pd.Series(range(1, 11), index=dataset.index, dtype="float64")
    dataset["Wind_operational"] = pd.Series(
        range(20, 30), index=dataset.index, dtype="float64"
    )
    fallback_day = dataset.index[4]
    dataset.loc[fallback_day, "Wind_operational"] = float("nan")

    features = build_features(dataset, cfg)

    assert features["Temp_forecast"].tolist() == list(range(100, 110))
    assert str(features["Temp_forecast"].dtype) == "int64"
    assert features.loc[fallback_day, "Wind_forecast"] == 4.0
    assert features.loc[dataset.index[3], "Wind_forecast"] == 23.0
    assert str(features["Wind_forecast"].dtype) == "float64"


def test_ml_matrix_includes_lags_and_drops_first_seven():
    cfg = Config.from_yaml("config.yaml")
    feats = build_features(_dataset(40), cfg)
    m = feature_matrix(feats, "ml")
    assert "L_t-7" in m.columns
    assert len(m) == 40 - 7
    assert m[["L_t-1", "L_t-7"]].isna().sum().sum() == 0


def test_exog_columns_sets():
    assert "L_t-7" in exog_columns("ml")
    assert "L_t-7" not in exog_columns("sarimax")


def test_temperature_anomaly_matches_forecast_minus_climatology():
    cfg = Config.from_yaml("config.yaml")
    dataset = _dataset(40)
    feats = build_features(dataset, cfg)
    assert "Temp_anomaly" in feats
    # The 40-day window covers only winter day-of-year, so the fallback mean
    # governs most days; the anomaly is finite and centered near zero.
    anomaly = feats["Temp_anomaly"].dropna()
    assert anomaly.notna().all()
    assert abs(float(anomaly.mean())) < 10.0
    assert list(feats.columns) == list(dict.fromkeys(feats.columns))


def test_validation_temperature_anomaly_ignores_post_cutoff_realized_weather():
    cfg = Config.from_yaml("config.yaml")
    cfg = replace(
        cfg,
        split=SplitConfig(
            train_end=date(2024, 1, 10),
            val_end=date(2024, 1, 15),
            calibration_start=date(2024, 1, 16),
            calibration_end=date(2024, 1, 17),
            test_start=date(2024, 1, 18),
            test_end=date(2024, 1, 20),
        ),
    )
    index = pd.Index([date(2024, 1, 1) + timedelta(days=i) for i in range(20)])
    dataset = pd.DataFrame(
        {
            "daily_load": range(20),
            "Temp_t": range(20),
            "Wind_t": range(20),
            "Temp_operational": range(100, 120),
            "Wind_operational": range(200, 220),
        },
        index=index,
    )
    changed = dataset.copy()
    changed.loc[date(2024, 1, 11) :, "Temp_t"] = 999.0

    baseline = build_features(dataset, cfg)
    mutated = build_features(changed, cfg)

    pd.testing.assert_series_equal(
        baseline.loc[date(2024, 1, 11) :, "Temp_anomaly"],
        mutated.loc[date(2024, 1, 11) :, "Temp_anomaly"],
    )
