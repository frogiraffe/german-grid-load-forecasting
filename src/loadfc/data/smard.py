"""Download SMARD filter 410: realised electricity consumption / grid load."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import requests

from .hourly import canonical_utc_index
from .http import resilient_session
from .validation import validate_load_series

_WEEK_MS = 7 * 24 * 3600 * 1000


def _index_url(cfg) -> str:
    s = cfg.smard
    return f"{s['base_url']}/{s['filter']}/{s['region']}/index_{s['resolution']}.json"


def _data_url(cfg, ts: int) -> str:
    s = cfg.smard
    return (
        f"{s['base_url']}/{s['filter']}/{s['region']}/"
        f"{s['filter']}_{s['region']}_{s['resolution']}_{ts}.json"
    )


def parse_series(payload: dict) -> pd.Series:
    points = payload["series"]
    stamps = pd.to_datetime([p[0] for p in points], unit="ms", utc=True)
    values = [p[1] for p in points]
    return pd.Series(values, index=stamps, dtype="float64").sort_index()


def to_daily(hourly: pd.Series) -> pd.Series:
    # SMARD timestamps are UTC instants; convert to German local time so the
    # daily mean covers the German calendar day, matching the weather series
    # (Open-Meteo is queried in Europe/Berlin local time).
    berlin = hourly.tz_convert("Europe/Berlin")
    daily = berlin.resample("D").mean()
    daily.index = daily.index.date
    daily.index.name = "date"
    daily.name = "daily_load"
    return validate_load_series(daily)


def to_hourly(series: pd.Series) -> pd.Series:
    """Aggregate SMARD sub-hourly values and validate a complete UTC timeline."""

    if not isinstance(series.index, pd.DatetimeIndex) or series.index.tz is None:
        raise ValueError("SMARD hourly data requires timezone-aware timestamps")
    hourly = series.tz_convert("UTC").resample("h").mean()
    hourly.name = "hourly_load"
    if hourly.isna().any():
        missing = hourly.index[hourly.isna()]
        sample = ", ".join(str(stamp) for stamp in missing[:3])
        raise ValueError(f"SMARD hourly load contains {len(missing)} empty hours (first: {sample})")
    validated = validate_load_series(hourly)
    return canonical_utc_index(validated)  # type: ignore[return-value]


def select_timestamps(week_starts: list[int], start: date, end: date) -> list[int]:
    lo = int(datetime(start.year, start.month, start.day, tzinfo=UTC).timestamp() * 1000)
    hi = (
        int(datetime(end.year, end.month, end.day, tzinfo=UTC).timestamp() * 1000)
        + 24 * 3600 * 1000
    )
    return [w for w in week_starts if (w + _WEEK_MS) > lo and w < hi]


def fetch_hourly_load(cfg, session: requests.Session | None = None) -> pd.Series:
    session = session or resilient_session()
    response = session.get(_index_url(cfg), timeout=60)
    response.raise_for_status()
    weeks = response.json()["timestamps"]
    weeks = select_timestamps(weeks, cfg.raw_start, cfg.raw_end)
    parts = []
    for ts in weeks:
        response = session.get(_data_url(cfg, ts), timeout=60)
        response.raise_for_status()
        payload = response.json()
        parts.append(parse_series(payload))
    if not parts:
        raise ValueError("SMARD returned no weekly series for the configured date range")
    full = pd.concat(parts).sort_index()
    return full[~full.index.duplicated(keep="last")]


def load_daily(cfg, session: requests.Session | None = None, refresh: bool = False) -> pd.Series:
    cache = cfg.path("raw_dir") / "load_daily.csv"
    if cache.exists() and not refresh:
        s = pd.read_csv(cache, index_col="date", parse_dates=["date"])["daily_load"]
        s.index = s.index.date
        s.index.name = "date"
        if s.index.min() <= cfg.raw_start and s.index.max() >= cfg.raw_end:
            return validate_load_series(s[(s.index >= cfg.raw_start) & (s.index <= cfg.raw_end)])
    raw = fetch_hourly_load(cfg, session)
    local_dates = raw.index.tz_convert("Europe/Berlin").date
    raw = raw[(local_dates >= cfg.raw_start) & (local_dates <= cfg.raw_end)]
    if raw.empty:
        raise ValueError("SMARD returned no daily load inside the configured range")
    daily = to_daily(raw)
    daily = daily[(daily.index >= cfg.raw_start) & (daily.index <= cfg.raw_end)]
    if daily.empty:
        raise ValueError("SMARD returned no daily load inside the configured range")
    cache.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(cache)
    return validate_load_series(daily)


def load_hourly(cfg, session: requests.Session | None = None, refresh: bool = False) -> pd.Series:
    """Load complete UTC-hourly load filtered by German local calendar date."""

    cache = cfg.path("raw_dir") / "load_hourly.parquet"
    if cache.exists() and not refresh:
        series = canonical_utc_index(pd.read_parquet(cache)["hourly_load"])
    else:
        raw = fetch_hourly_load(cfg, session)
        local_dates = raw.index.tz_convert("Europe/Berlin").date
        raw = raw[(local_dates >= cfg.raw_start) & (local_dates <= cfg.raw_end)]
        if raw.empty:
            raise ValueError("SMARD returned no hourly load inside the configured range")
        series = to_hourly(raw)
        cache.parent.mkdir(parents=True, exist_ok=True)
        series.to_frame().to_parquet(cache)
    local_dates = series.index.tz_convert("Europe/Berlin").date
    selected = series[(local_dates >= cfg.raw_start) & (local_dates <= cfg.raw_end)]
    if selected.empty:
        raise ValueError("SMARD returned no hourly load inside the configured range")
    return validate_load_series(selected)
