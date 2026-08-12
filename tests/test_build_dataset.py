from datetime import date

import pandas as pd
import pytest

from loadfc.data.build_dataset import assemble, assemble_hourly


def _frames():
    days = pd.to_datetime(pd.date_range("2019-01-10", "2019-01-20")).date
    load = pd.Series(
        range(50000, 50000 + len(days)), index=days, name="daily_load", dtype="float64"
    )
    weather = pd.DataFrame({"Temp_t": [5.0] * len(days), "Wind_t": [10.0] * len(days)}, index=days)
    return load, weather


def test_assemble_joins_and_trims_to_dataset_start():
    load, weather = _frames()
    df = assemble(load, weather, dataset_start=date(2019, 1, 14))
    assert df.index.min() == date(2019, 1, 14)
    assert list(df.columns) == ["daily_load", "Temp_t", "Wind_t"]
    assert df.loc[date(2019, 1, 14), "Temp_t"] == 5.0


def test_assemble_drops_rows_with_missing_load():
    load, weather = _frames()
    load.loc[date(2019, 1, 15)] = float("nan")
    df = assemble(load, weather, dataset_start=date(2019, 1, 14))
    assert date(2019, 1, 15) not in df.index


def test_assemble_requires_overlap():
    load, weather = _frames()
    with pytest.raises(ValueError, match="overlap"):
        assemble(load, weather, dataset_start=date(2025, 1, 1))


def test_assemble_preserves_operational_weather_columns():
    load, weather = _frames()
    operational = pd.DataFrame(
        {
            "Temp_operational": [7.0] * 3,
            "Wind_operational": [12.0] * 3,
        },
        index=[date(2019, 1, 14), date(2019, 1, 15), date(2019, 1, 16)],
    )
    frame = assemble(load, weather, date(2019, 1, 14), operational)
    assert {"Temp_operational", "Wind_operational"} <= set(frame)
    assert frame.loc[date(2019, 1, 14), "Temp_operational"] == 7.0
    assert pd.isna(frame.loc[date(2019, 1, 17), "Temp_operational"])


def test_build_writes_parquet(tmp_path, monkeypatch):
    from loadfc.config import Config
    from loadfc.data import build_dataset

    days = pd.to_datetime(pd.date_range("2019-01-10", "2019-01-20")).date
    load = pd.Series(
        range(50000, 50000 + len(days)), index=days, name="daily_load", dtype="float64"
    )
    weather = pd.DataFrame({"Temp_t": [5.0] * len(days), "Wind_t": [10.0] * len(days)}, index=days)

    monkeypatch.setattr(build_dataset.smard, "load_daily", lambda cfg, refresh=False: load)
    monkeypatch.setattr(
        build_dataset.weather_mod, "weather_daily", lambda cfg, refresh=False: weather
    )
    monkeypatch.setattr(
        build_dataset.weather_mod,
        "operational_weather_daily",
        lambda cfg, refresh=False: pd.DataFrame(
            {
                "Temp_operational": [5.0] * len(days),
                "Wind_operational": [10.0] * len(days),
            },
            index=days,
        ),
    )
    monkeypatch.setattr(Config, "path", lambda self, key: tmp_path / key)

    cfg = Config.from_yaml("config.yaml")
    df = build_dataset.build(cfg)
    assert (tmp_path / "processed_dir" / "dataset.parquet").exists()
    assert len(df) > 0


def test_assemble_hourly_joins_on_utc_and_trims_by_german_date():
    index = pd.date_range("2019-01-13 22:00", periods=5, freq="h", tz="UTC")
    load = pd.Series(range(5), index=index, dtype="float64")
    weather = pd.DataFrame({"Temp_t": [5.0] * 5, "Wind_t": [10.0] * 5}, index=index)

    frame = assemble_hourly(load, weather, date(2019, 1, 14))

    assert frame.index.min() == pd.Timestamp("2019-01-13 23:00", tz="UTC")
    assert list(frame.columns) == ["hourly_load", "Temp_t", "Wind_t"]
    assert frame.index.name == "timestamp_utc"


def test_assemble_hourly_does_not_backfill_unavailable_forecasts():
    index = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    load = pd.Series([1.0, 2.0, 3.0], index=index)
    weather = pd.DataFrame({"Temp_t": [4.0] * 3, "Wind_t": [8.0] * 3}, index=index)
    operational = pd.DataFrame(
        {"Temp_operational": [5.0], "Wind_operational": [9.0]},
        index=index[1:2],
    )

    frame = assemble_hourly(load, weather, date(2024, 1, 1), operational)

    assert pd.isna(frame.iloc[0]["Temp_operational"])
    assert frame.iloc[1]["Temp_operational"] == 5.0


def test_build_hourly_writes_separate_artifacts(tmp_path, monkeypatch):
    from loadfc.config import Config
    from loadfc.data import build_dataset

    index = pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC")
    load = pd.Series([1.0, 2.0, 3.0], index=index)
    observed = pd.DataFrame({"Temp_t": [4.0] * 3, "Wind_t": [8.0] * 3}, index=index)
    operational = pd.DataFrame(
        {"Temp_operational": [5.0] * 3, "Wind_operational": [9.0] * 3},
        index=index,
    )
    monkeypatch.setattr(build_dataset.smard, "load_hourly", lambda cfg, refresh=False: load)
    monkeypatch.setattr(
        build_dataset.weather_mod,
        "weather_hourly",
        lambda cfg, refresh=False: observed,
    )
    monkeypatch.setattr(
        build_dataset.weather_mod,
        "operational_weather_hourly",
        lambda cfg, refresh=False: operational,
    )
    monkeypatch.setattr(Config, "path", lambda self, key: tmp_path / key)

    frame = build_dataset.build_hourly(Config.from_yaml("config.yaml"))

    assert len(frame) == 3
    assert (tmp_path / "processed_dir" / "dataset_hourly.parquet").exists()
    assert (tmp_path / "processed_dir" / "dataset_hourly.csv").exists()
