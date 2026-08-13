from dataclasses import replace
from datetime import date

import pandas as pd
import pytest

from loadfc.config import Config
from loadfc.evaluation.conformal import adaptive_conformal_interval, split_conformal_interval
from loadfc.evaluation.provenance import (
    artifact_results_root,
    daily_prediction_artifact,
    hourly_prediction_artifact,
)
from loadfc.features.assemble import feature_matrix
from scripts.run_intervals import interval_prediction_frame


def test_daily_prediction_artifact_is_point_in_time_auditable(tmp_path):
    valid_time = date(2024, 1, 2)
    predictions = pd.DataFrame({"actual": [100.0], "forecast": [99.0]}, index=[valid_time])
    weather = pd.DataFrame(
        {"Temp_operational": [3.0], "Wind_operational": [8.0]}, index=[valid_time]
    )

    artifact = daily_prediction_artifact(
        predictions,
        weather,
        weather_strategy="available_day_ahead",
    )
    path = tmp_path / "prediction.csv"
    artifact.to_csv(path)
    readback = pd.read_csv(path, index_col=0)

    assert list(readback.columns) == [
        "actual",
        "forecast",
        "forecast_origin",
        "valid_time",
        "weather_source_run",
        "weather_availability_assumption",
    ]
    assert readback.loc[str(valid_time), "forecast_origin"] == "2024-01-01"
    assert readback.loc[str(valid_time), "weather_source_run"] == "open_meteo_previous_day1"
    assert "exact provider run timestamp unavailable" in readback.loc[
        str(valid_time), "weather_availability_assumption"
    ]


def test_daily_writer_keeps_provenance_out_of_model_features():
    features = pd.DataFrame(
        {
            "daily_load": [100.0, 101.0],
            "Weekend": [0, 0],
            "holiday_national": [0, 0],
            "holiday_religious": [0, 0],
            "bridge_day": [0, 0],
            "pre_holiday": [0, 0],
            "post_holiday": [0, 0],
            "Temp_forecast": [5.0, 6.0],
            "Wind_forecast": [7.0, 8.0],
            "HDD": [13.0, 12.0],
            "CDD": [0.0, 0.0],
            "Temp_anomaly": [1.0, 1.0],
            "sin_1_week": [0.0, 0.0],
            "cos_1_week": [1.0, 1.0],
            "sin_1_year": [0.0, 0.0],
            "cos_1_year": [1.0, 1.0],
            "L_t-1": [99.0, 100.0],
            "L_t-7": [95.0, 96.0],
        }
    )
    features["forecast_origin"] = date(2024, 1, 1)
    features["valid_time"] = date(2024, 1, 2)
    features["weather_source_run"] = "open_meteo_previous_day1"
    features["weather_availability_assumption"] = "previous_day1"

    matrix = feature_matrix(features, "ml")

    assert not {
        "forecast_origin",
        "valid_time",
        "weather_source_run",
        "weather_availability_assumption",
    } & set(matrix.columns)


@pytest.mark.parametrize(
    ("temp_operational", "wind_operational", "source"),
    [
        (3.0, 8.0, "open_meteo_previous_day1"),
        (3.0, None, "open_meteo_previous_day1_temperature_persistence_wind"),
        (None, 8.0, "persistence_temperature_open_meteo_previous_day1_wind"),
        (None, None, "persistence"),
    ],
)
def test_available_day_ahead_labels_match_operational_availability(
    temp_operational, wind_operational, source
):
    valid_time = date(2024, 1, 2)
    artifact = daily_prediction_artifact(
        pd.DataFrame({"actual": [100.0], "forecast": [99.0]}, index=[valid_time]),
        pd.DataFrame(
            {"Temp_operational": [temp_operational], "Wind_operational": [wind_operational]},
            index=[valid_time],
        ),
        weather_strategy="available_day_ahead",
    )

    assert artifact.iloc[0]["weather_source_run"] == source


def test_artifact_roots_and_daily_identity_validation():
    cfg = Config.from_yaml("config.yaml")
    oracle_cfg = replace(cfg, features={**cfg.features, "weather_strategy": "oracle"})
    assert artifact_results_root(cfg) == cfg.path("results_dir")
    assert artifact_results_root(oracle_cfg) == cfg.path("results_dir") / "sensitivity" / "oracle"

    weather = pd.DataFrame(
        {"Temp_operational": [3.0, 4.0], "Wind_operational": [8.0, 9.0]},
        index=[date(2024, 1, 1), date(2024, 1, 2)],
    )
    with pytest.raises(ValueError):
        daily_prediction_artifact(pd.DataFrame(columns=["actual", "forecast"]), weather, weather_strategy="available_day_ahead")
    with pytest.raises(ValueError):
        daily_prediction_artifact(
            pd.DataFrame({"actual": [1.0], "forecast": [1.0]}, index=[pd.NaT]),
            weather,
            weather_strategy="available_day_ahead",
        )
    with pytest.raises(ValueError):
        daily_prediction_artifact(
            pd.DataFrame({"actual": [1.0, 2.0], "forecast": [1.0, 2.0]}, index=[date(2024, 1, 1)] * 2),
            weather,
            weather_strategy="available_day_ahead",
        )

    artifact = daily_prediction_artifact(
        pd.DataFrame(
            {"actual": [2.0, 1.0], "forecast": [2.0, 1.0]},
            index=[date(2024, 1, 2), date(2024, 1, 1)],
        ),
        weather,
        weather_strategy="available_day_ahead",
    )
    assert list(artifact.index) == [date(2024, 1, 1), date(2024, 1, 2)]


def test_daily_intervals_retain_prediction_provenance():
    calibration_actual = pd.Series([10.0, 12.0, 14.0])
    calibration_forecast = pd.Series([9.0, 11.0, 13.0])
    test = pd.DataFrame(
        {
            "actual": [16.0, 18.0],
            "forecast": [15.0, 19.0],
            "forecast_origin": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
            "valid_time": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
            "weather_source_run": ["open_meteo_previous_day1", "persistence"],
                "weather_availability_assumption": ["previous_day1", "previous-day realized"],
                "evaluation_period": ["retrospective_final", "retrospective_final"],
                "stream_id": ["daily/test", "daily/test"],
                "protocol_fingerprint": ["abc", "abc"],
                "point_state_policy": ["{}", "{}"],
                "interval_state_policy": ["{}", "{}"],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )

    frame = interval_prediction_frame(test)
    fixed_lower, fixed_upper = split_conformal_interval(
        calibration_actual, calibration_forecast, test["forecast"], 0.2
    )
    adaptive_lower, adaptive_upper, _ = adaptive_conformal_interval(
        calibration_actual,
        calibration_forecast,
        test["actual"],
        test["forecast"],
        0.2,
        gamma=0.1,
        window=3,
    )
    frame["fixed_lower_80"] = fixed_lower
    frame["fixed_upper_80"] = fixed_upper
    frame["lower_80"] = adaptive_lower
    frame["upper_80"] = adaptive_upper

    assert frame[["actual", "forecast"]].equals(test[["actual", "forecast"]])
    assert frame["forecast_origin"].equals(test["forecast_origin"])
    assert frame["valid_time"].equals(test["valid_time"])
    assert frame["weather_source_run"].equals(test["weather_source_run"])
    assert frame["weather_availability_assumption"].equals(
        test["weather_availability_assumption"]
    )
    assert list(frame["fixed_lower_80"]) == list(fixed_lower)
    assert list(frame["upper_80"]) == list(adaptive_upper)


def test_hourly_artifact_retains_interval_columns():
    valid_time = pd.date_range("2024-01-02", periods=1, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "actual": [100.0],
            "forecast": [99.0],
            "forecast_origin": [valid_time[0] - pd.Timedelta(hours=24)],
            "valid_time": valid_time,
            "horizon": [24],
            "lower": [90.0],
            "upper": [110.0],
        }
    ).set_index(["forecast_origin", "valid_time", "horizon"], drop=False)

    artifact = hourly_prediction_artifact(frame, pd.DataFrame(), weather_strategy="persistence")

    assert artifact[["lower", "upper"]].iloc[0].tolist() == [90.0, 110.0]


def test_hourly_artifact_rejects_valid_time_that_does_not_match_horizon():
    valid_time = pd.Timestamp("2024-01-02T00:00:00Z")
    frame = pd.DataFrame(
        {
            "actual": [100.0],
            "forecast": [99.0],
            "forecast_origin": [valid_time - pd.Timedelta(hours=12)],
            "valid_time": [valid_time],
            "horizon": [24],
        }
    ).set_index(["forecast_origin", "valid_time", "horizon"], drop=False)

    with pytest.raises(ValueError, match="horizon"):
        hourly_prediction_artifact(frame, pd.DataFrame(), weather_strategy="persistence")
