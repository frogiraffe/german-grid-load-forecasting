import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from loadfc.data import smard

FX = Path(__file__).parent / "fixtures"


def test_parse_series_builds_utc_indexed_series():
    payload = json.loads((FX / "smard_week.json").read_text())
    s = smard.parse_series(payload)
    assert isinstance(s.index, pd.DatetimeIndex)
    assert len(s) == 3
    assert s.iloc[0] == 50000.0
    assert pd.isna(s.iloc[1])  # null preserved as NaN


def test_to_daily_means_over_day_ignoring_nan():
    idx = pd.to_datetime(["2019-01-01 06:00", "2019-01-01 07:00", "2019-01-02 06:00"], utc=True)
    s = pd.Series([100.0, float("nan"), 300.0], index=idx)
    daily = smard.to_daily(s)
    assert daily.loc[date(2019, 1, 1)] == 100.0  # mean skips the NaN hour
    assert daily.loc[date(2019, 1, 2)] == 300.0


def test_to_daily_bins_by_german_local_day():
    # 2019-01-01 23:00 UTC == 2019-01-02 00:00 CET, so it belongs to Jan 2 in
    # German local time even though it is still Jan 1 in UTC.
    idx = pd.to_datetime(["2019-01-01 22:00", "2019-01-01 23:00"], utc=True)
    s = pd.Series([100.0, 200.0], index=idx)
    daily = smard.to_daily(s)
    assert daily.loc[date(2019, 1, 1)] == 100.0  # 23:00 CET -> Jan 1
    assert daily.loc[date(2019, 1, 2)] == 200.0  # 00:00 CET -> Jan 2


def test_to_hourly_aggregates_quarter_hours_on_a_utc_timeline():
    index = pd.date_range("2024-01-01", periods=8, freq="15min", tz="UTC")
    source = pd.Series(range(8), index=index, dtype="float64")

    hourly = smard.to_hourly(source)

    assert list(hourly) == [1.5, 5.5]
    assert str(hourly.index.tz) == "UTC"
    assert hourly.index.name == "timestamp_utc"


def test_to_hourly_rejects_an_empty_hour():
    index = pd.to_datetime(["2024-01-01T00:00Z", "2024-01-01T02:00Z"])
    with pytest.raises(ValueError, match="empty hours"):
        smard.to_hourly(pd.Series([1.0, 2.0], index=index))


def test_select_timestamps_keeps_weeks_overlapping_range():
    # weekly-start epochs (ms): 2019-01-01 and 2019-01-08
    weeks = [1546297200000, 1546902000000]
    kept = smard.select_timestamps(weeks, date(2019, 1, 9), date(2019, 1, 10))
    assert kept == [1546902000000]


class _Response:
    def __init__(self, payload):
        self.payload = payload
        self.checked = False

    def raise_for_status(self):
        self.checked = True

    def json(self):
        return self.payload


class _Session:
    def __init__(self, payloads):
        self.payloads = iter(payloads)
        self.urls = []
        self.responses = []

    def get(self, url, **kwargs):
        self.urls.append((url, kwargs))
        response = _Response(next(self.payloads))
        self.responses.append(response)
        return response


def _cfg(tmp_path, start=date(2019, 1, 1), end=date(2019, 1, 1)):
    return SimpleNamespace(
        raw_start=start,
        raw_end=end,
        smard={
            "base_url": "https://smard.example",
            "filter": 410,
            "region": "DE",
            "resolution": "hour",
        },
        path=lambda key: tmp_path,
    )


def test_fetch_hourly_load_requests_index_and_selected_week(tmp_path):
    week = 1546297200000
    session = _Session(
        [
            {"timestamps": [week]},
            {"series": [[1546297200000, 50000.0], [1546300800000, 51000.0]]},
        ]
    )
    series = smard.fetch_hourly_load(_cfg(tmp_path), session)
    assert len(series) == 2
    assert session.urls[0][0].endswith("/410/DE/index_hour.json")
    assert session.urls[1][0].endswith(f"/410_DE_hour_{week}.json")
    assert all(response.checked for response in session.responses)


def test_fetch_hourly_load_rejects_empty_selection(tmp_path):
    session = _Session([{"timestamps": []}])
    with pytest.raises(ValueError, match="no weekly"):
        smard.fetch_hourly_load(_cfg(tmp_path), session)


def test_load_daily_writes_and_reuses_cache(tmp_path):
    index = pd.to_datetime(["2019-01-01 00:00", "2019-01-01 12:00"], utc=True)
    hourly = pd.Series([50000.0, 52000.0], index=index)
    original = smard.fetch_hourly_load
    smard.fetch_hourly_load = lambda cfg, session=None: hourly
    try:
        first = smard.load_daily(_cfg(tmp_path), refresh=True)
        second = smard.load_daily(_cfg(tmp_path))
    finally:
        smard.fetch_hourly_load = original
    assert first.iloc[0] == 51000.0
    pd.testing.assert_series_equal(first, second)


def test_load_daily_rejects_data_outside_requested_range(tmp_path):
    index = pd.to_datetime(["2018-01-01 12:00"], utc=True)
    original = smard.fetch_hourly_load
    smard.fetch_hourly_load = lambda cfg, session=None: pd.Series([1.0], index=index)
    try:
        with pytest.raises(ValueError, match="inside"):
            smard.load_daily(_cfg(tmp_path), refresh=True)
    finally:
        smard.fetch_hourly_load = original


def test_load_hourly_reuses_timezone_preserving_parquet_cache(tmp_path):
    index = pd.date_range("2018-12-31 23:00", periods=24, freq="h", tz="UTC")
    source = pd.Series(range(1, 25), index=index, dtype="float64")
    original = smard.fetch_hourly_load
    smard.fetch_hourly_load = lambda cfg, session=None: source
    try:
        first = smard.load_hourly(_cfg(tmp_path), refresh=True)
        second = smard.load_hourly(_cfg(tmp_path))
    finally:
        smard.fetch_hourly_load = original
    pd.testing.assert_series_equal(first, second, check_freq=False)
    assert first.index.min() == pd.Timestamp("2018-12-31 23:00", tz="UTC")


def test_load_hourly_ignores_nulls_after_requested_local_date(tmp_path):
    index = pd.date_range("2018-12-31 23:00", periods=26, freq="h", tz="UTC")
    values = [float(value) for value in range(1, 25)] + [float("nan"), float("nan")]
    original = smard.fetch_hourly_load
    smard.fetch_hourly_load = lambda cfg, session=None: pd.Series(values, index=index)
    try:
        result = smard.load_hourly(_cfg(tmp_path), refresh=True)
    finally:
        smard.fetch_hourly_load = original

    assert len(result) == 24
    assert not result.isna().any()
