import json
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from loadfc.config import City
from loadfc.data import weather

FX = Path(__file__).parent / "fixtures"


def test_parse_open_meteo_indexes_by_time():
    payload = json.loads((FX / "open_meteo_city.json").read_text())
    df = weather.parse_open_meteo(payload, ["temperature_2m", "wind_speed_10m"])
    assert list(df.columns) == ["temperature_2m", "wind_speed_10m"]
    assert len(df) == 4


def test_daily_city_means_per_day():
    payload = json.loads((FX / "open_meteo_city.json").read_text())
    df = weather.parse_open_meteo(payload, ["temperature_2m", "wind_speed_10m"])
    daily = weather.daily_city(df)
    assert daily.loc[date(2019, 1, 1), "temperature_2m"] == 5.0  # mean(0,10)
    assert daily.loc[date(2019, 1, 2), "wind_speed_10m"] == 10.0  # mean(9,11)


def test_canonical_utc_index_disambiguates_fall_dst_by_utc_instant():
    local = pd.to_datetime(
        [
            "2024-10-27 01:00",
            "2024-10-27 02:00",
            "2024-10-27 02:00",
            "2024-10-27 03:00",
        ]
    )
    frame = pd.DataFrame({"temperature_2m": range(4)}, index=local)

    hourly = weather.canonical_utc_index(frame, source_timezone="Europe/Berlin")

    assert str(hourly.index.tz) == "UTC"
    assert hourly.index.is_unique
    assert len(hourly) == 4


def test_canonical_utc_index_rejects_missing_utc_instants():
    local = pd.to_datetime(["2024-01-01 00:00", "2024-01-01 02:00"])
    frame = pd.DataFrame({"temperature_2m": [1.0, 2.0]}, index=local)

    with pytest.raises(ValueError, match="missing 1 UTC"):
        weather.canonical_utc_index(frame, source_timezone="Europe/Berlin")


def test_canonical_utc_index_rejects_off_grid_instants():
    index = pd.to_datetime(
        [
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:30:00Z",
            "2024-01-01T01:00:00Z",
        ]
    )
    frame = pd.DataFrame({"temperature_2m": [1.0, 2.0, 3.0]}, index=index)

    with pytest.raises(ValueError, match="off-grid"):
        weather.canonical_utc_index(frame, source_timezone="UTC")


def test_canonical_utc_index_rejects_empty_input():
    frame = pd.DataFrame({"temperature_2m": []}, index=pd.DatetimeIndex([], dtype="datetime64[ns]"))
    with pytest.raises(ValueError, match="must not be empty"):
        weather.canonical_utc_index(frame, source_timezone="Europe/Berlin")


@pytest.mark.parametrize(
    ("local_day", "expected_hours"),
    [("2024-03-31", 23), ("2024-01-02", 24), ("2024-10-27", 25)],
)
def test_canonical_utc_index_preserves_complete_berlin_local_days(local_day, expected_hours):
    start = pd.Timestamp(local_day, tz="Europe/Berlin")
    end = pd.Timestamp(start.date() + timedelta(days=1), tz="Europe/Berlin")
    index = pd.date_range(start, end, freq="h", inclusive="left")
    frame = pd.DataFrame({"temperature_2m": range(len(index))}, index=index.tz_convert("UTC"))

    hourly = weather.canonical_utc_index(frame)

    assert len(hourly) == expected_hours
    assert hourly.index.is_unique


def test_combine_pop_weighted():
    idx = [date(2019, 1, 1)]
    a = pd.DataFrame({"temperature_2m": [10.0], "wind_speed_10m": [4.0]}, index=idx)
    b = pd.DataFrame({"temperature_2m": [20.0], "wind_speed_10m": [8.0]}, index=idx)
    out = weather.combine_pop_weighted({"A": a, "B": b}, {"A": 0.25, "B": 0.75})
    assert out.loc[date(2019, 1, 1), "Temp_t"] == pytest.approx(17.5)
    assert out.loc[date(2019, 1, 1), "Wind_t"] == pytest.approx(7.0)


def test_combine_pop_weighted_rejects_empty_input():
    with pytest.raises(ValueError, match="empty"):
        weather.combine_pop_weighted({}, {})


def test_date_chunks_are_contiguous_and_bounded():
    chunks = list(weather._date_chunks(date(2024, 1, 1), date(2024, 1, 5), days=2))
    assert chunks == [
        (date(2024, 1, 1), date(2024, 1, 2)),
        (date(2024, 1, 3), date(2024, 1, 4)),
        (date(2024, 1, 5), date(2024, 1, 5)),
    ]


class _Response:
    def __init__(self, payload):
        self._payload = payload
        self.status_checked = False

    def raise_for_status(self):
        self.status_checked = True

    def json(self):
        return self._payload


class _Session:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []
        self.responses = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = _Response(self.payload)
        self.responses.append(response)
        return response


def _cfg(tmp_path, *, operational=False):
    variables = (
        ["temperature_2m_previous_day1", "wind_speed_10m_previous_day1"]
        if operational
        else ["temperature_2m", "wind_speed_10m"]
    )
    return SimpleNamespace(
        raw_start=date(2024, 1, 1),
        raw_end=date(2024, 1, 1),
        weather={
            "base_url": "https://archive.example",
            "previous_runs_url": "https://runs.example",
            "operational_start": "2024-01-01",
            "hourly_vars": ["temperature_2m", "wind_speed_10m"],
            "operational_hourly_vars": variables,
            "timezone": "Europe/Berlin",
        },
        cities=[
            City("A", 1.0, 2.0, 1),
            City("B", 3.0, 4.0, 3),
        ],
        city_weights=lambda: {"A": 0.25, "B": 0.75},
        path=lambda key: tmp_path,
    )


def test_weather_daily_fetches_cities_and_writes_cache(tmp_path):
    payload = {
        "hourly": {
            "time": ["2024-01-01T00:00", "2024-01-01T12:00"],
            "temperature_2m": [2.0, 6.0],
            "wind_speed_10m": [8.0, 12.0],
        }
    }
    session = _Session(payload)
    out = weather.weather_daily(_cfg(tmp_path), session=session, refresh=True)

    assert len(session.calls) == 2
    assert all(response.status_checked for response in session.responses)
    assert out.loc[date(2024, 1, 1), "Temp_t"] == 4.0
    assert (tmp_path / "weather_daily.csv").exists()


def test_weather_daily_uses_complete_cache(tmp_path):
    pd.DataFrame(
        {"Temp_t": [4.0], "Wind_t": [10.0]},
        index=pd.Index([date(2024, 1, 1)], name="date"),
    ).to_csv(tmp_path / "weather_daily.csv")
    out = weather.weather_daily(_cfg(tmp_path), session=None)
    assert out.iloc[0]["Temp_t"] == 4.0


def test_weather_hourly_fetches_utc_data_and_reuses_parquet_cache(tmp_path):
    payload = {
        "hourly": {
            "time": ["2024-01-01T00:00", "2024-01-01T01:00"],
            "temperature_2m": [2.0, 6.0],
            "wind_speed_10m": [8.0, 12.0],
        }
    }
    session = _Session(payload)

    first = weather.weather_hourly(_cfg(tmp_path), session=session, refresh=True)
    second = weather.weather_hourly(_cfg(tmp_path))

    assert len(session.calls) == 2
    assert all(call[1]["params"]["timezone"] == "UTC" for call in session.calls)
    assert str(first.index.tz) == "UTC"
    assert first.index.name == "timestamp_utc"
    pd.testing.assert_frame_equal(first, second, check_freq=False)
    assert (tmp_path / "weather_hourly.parquet").exists()


def test_operational_weather_uses_previous_day1_fields(tmp_path):
    payload = {
        "hourly": {
            "time": ["2024-01-01T00:00", "2024-01-01T12:00"],
            "temperature_2m_previous_day1": [1.0, 5.0],
            "wind_speed_10m_previous_day1": [6.0, 10.0],
        }
    }
    session = _Session(payload)
    out = weather.operational_weather_daily(
        _cfg(tmp_path, operational=True), session=session, refresh=True
    )

    assert {call[0] for call in session.calls} == {"https://runs.example"}
    assert list(out.columns) == ["Temp_operational", "Wind_operational"]
    assert out.iloc[0]["Temp_operational"] == 3.0


def test_operational_weather_rejects_incomplete_days(tmp_path):
    payload = {
        "hourly": {
            "time": ["2024-01-01T00:00"],
            "temperature_2m_previous_day1": [None],
            "wind_speed_10m_previous_day1": [8.0],
        }
    }
    with pytest.raises(ValueError, match="incomplete"):
        weather.operational_weather_daily(
            _cfg(tmp_path, operational=True),
            session=_Session(payload),
            refresh=True,
        )
