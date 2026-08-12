"""Open-Meteo observations and archived day-ahead weather forecasts."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import requests

from .hourly import canonical_utc_index
from .http import resilient_session
from .validation import validate_weather_frame

_RENAME = {"temperature_2m": "Temp_t", "wind_speed_10m": "Wind_t"}
_OPERATIONAL_RENAME = {
    "temperature_2m_previous_day1": "Temp_operational",
    "wind_speed_10m_previous_day1": "Wind_operational",
}


def parse_open_meteo(payload: dict, hourly_vars: list[str]) -> pd.DataFrame:
    # Open-Meteo is queried with timezone=Europe/Berlin, so these timestamps are
    # naive German local time; keeping them naive bins the daily mean by the
    # German calendar day (consistent with the load series).
    h = payload["hourly"]
    idx = pd.to_datetime(h["time"])
    frame = pd.DataFrame({v: h[v] for v in hourly_vars}, index=idx)
    return validate_weather_frame(frame, hourly_vars, nullable=True)


def daily_city(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.resample("D").mean()
    daily.index = daily.index.date
    daily.index.name = "date"
    return daily


def _hourly_cache_slice(
    path,
    *,
    start: date,
    end: date,
    timezone: str,
) -> pd.DataFrame | None:
    if not path.exists():
        return None
    frame = canonical_utc_index(pd.read_parquet(path), source_timezone=timezone)  # type: ignore[return-value]
    local_dates = frame.index.tz_convert(timezone).date
    if local_dates.min() > start or local_dates.max() < end:
        return None
    return frame[(local_dates >= start) & (local_dates <= end)]


def combine_pop_weighted(
    city_daily: dict[str, pd.DataFrame],
    weights: dict[str, float],
    rename: dict[str, str] | None = None,
) -> pd.DataFrame:
    if not city_daily:
        raise ValueError("city_daily is empty; need at least one city")
    acc: pd.DataFrame | None = None
    for name, df in city_daily.items():
        contribution = df * weights[name]
        acc = contribution if acc is None else acc.add(contribution)
    assert acc is not None
    return acc.rename(columns=rename or _RENAME)


def _fetch_city_frame(
    cfg,
    city,
    session: requests.Session,
    hourly_vars: list[str],
    *,
    base_url: str | None = None,
    start: date | None = None,
    end: date | None = None,
    timezone: str | None = None,
) -> pd.DataFrame:
    params = {
        "latitude": city.lat,
        "longitude": city.lon,
        "start_date": (start or cfg.raw_start).isoformat(),
        "end_date": (end or cfg.raw_end).isoformat(),
        "hourly": ",".join(hourly_vars),
        "timezone": timezone or cfg.weather["timezone"],
    }
    response = session.get(base_url or cfg.weather["base_url"], params=params, timeout=120)
    response.raise_for_status()
    payload = response.json()
    return parse_open_meteo(payload, hourly_vars)


def _fetch_city_hourly(
    cfg,
    city,
    session: requests.Session,
    hourly_vars: list[str],
    *,
    base_url: str | None = None,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    return daily_city(
        _fetch_city_frame(
            cfg,
            city,
            session,
            hourly_vars,
            base_url=base_url,
            start=start,
            end=end,
        )
    )


def _fetch_city_utc_hourly(
    cfg,
    city,
    session: requests.Session,
    hourly_vars: list[str],
    *,
    base_url: str | None = None,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    frame = _fetch_city_frame(
        cfg,
        city,
        session,
        hourly_vars,
        base_url=base_url,
        start=start,
        end=end,
        timezone="UTC",
    )
    return canonical_utc_index(frame, source_timezone="UTC")  # type: ignore[return-value]


def _date_chunks(start: date, end: date, days: int = 90):
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=days - 1), end)
        yield cursor, chunk_end
        cursor = chunk_end + timedelta(days=1)


def weather_daily(
    cfg, session: requests.Session | None = None, refresh: bool = False
) -> pd.DataFrame:
    cache = cfg.path("raw_dir") / "weather_daily.csv"
    if cache.exists() and not refresh:
        df = pd.read_csv(cache, index_col="date", parse_dates=["date"])
        df.index = df.index.date
        df.index.name = "date"
        if df.index.min() <= cfg.raw_start and df.index.max() >= cfg.raw_end:
            return validate_weather_frame(
                df[(df.index >= cfg.raw_start) & (df.index <= cfg.raw_end)],
                ["Temp_t", "Wind_t"],
            )
    session = session or resilient_session()
    city_daily = {
        c.name: _fetch_city_hourly(cfg, c, session, cfg.weather["hourly_vars"]) for c in cfg.cities
    }
    out = combine_pop_weighted(city_daily, cfg.city_weights())
    out = validate_weather_frame(out, ["Temp_t", "Wind_t"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache)
    return out


def weather_hourly(
    cfg, session: requests.Session | None = None, refresh: bool = False
) -> pd.DataFrame:
    """Return population-weighted observations on a UTC-hourly index."""

    cache = cfg.path("raw_dir") / "weather_hourly.parquet"
    if not refresh:
        cached = _hourly_cache_slice(
            cache,
            start=cfg.raw_start,
            end=cfg.raw_end,
            timezone=cfg.weather["timezone"],
        )
        if cached is not None:
            return validate_weather_frame(cached, ["Temp_t", "Wind_t"])
    session = session or resilient_session()
    city_hourly = {
        city.name: _fetch_city_utc_hourly(cfg, city, session, cfg.weather["hourly_vars"])
        for city in cfg.cities
    }
    out = combine_pop_weighted(city_hourly, cfg.city_weights())
    if out.isna().any().any():
        raise ValueError("hourly weather timelines do not align across cities")
    out = validate_weather_frame(out, ["Temp_t", "Wind_t"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache)
    return out


def operational_weather_daily(
    cfg,
    session: requests.Session | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return forecasts issued 24 hours before each valid timestamp.

    Open-Meteo's Previous Runs API aligns the ``*_previous_day1`` variables to
    valid time. Daily means therefore describe the forecast available one day
    before the load being predicted.
    """
    cache = cfg.path("raw_dir") / "weather_operational_daily.csv"
    start = pd.Timestamp(cfg.weather["operational_start"]).date()
    end = cfg.raw_end
    if cache.exists() and not refresh:
        frame = pd.read_csv(cache, index_col="date", parse_dates=["date"])
        frame.index = frame.index.date
        frame.index.name = "date"
        if frame.index.min() <= start and frame.index.max() >= end:
            return validate_weather_frame(
                frame[(frame.index >= start) & (frame.index <= end)],
                ["Temp_operational", "Wind_operational"],
            )

    session = session or resilient_session()
    variables = list(cfg.weather["operational_hourly_vars"])
    city_daily = {}
    for city in cfg.cities:
        parts = [
            _fetch_city_hourly(
                cfg,
                city,
                session,
                variables,
                base_url=cfg.weather["previous_runs_url"],
                start=chunk_start,
                end=chunk_end,
            )
            for chunk_start, chunk_end in _date_chunks(start, end)
        ]
        city_daily[city.name] = pd.concat(parts).sort_index()
    out = combine_pop_weighted(city_daily, cfg.city_weights(), _OPERATIONAL_RENAME)
    if out.isna().any().any():
        missing_index = out.index[out.isna().any(axis=1)]
        sample = ", ".join(str(day) for day in missing_index[:5])
        raise ValueError(
            f"operational weather contains {len(missing_index)} incomplete days (first: {sample})"
        )
    out = validate_weather_frame(
        out,
        ["Temp_operational", "Wind_operational"],
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache)
    return out


def operational_weather_hourly(
    cfg,
    session: requests.Session | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return forecasts issued 24 hours before each hourly valid instant."""

    cache = cfg.path("raw_dir") / "weather_operational_hourly.parquet"
    start = pd.Timestamp(cfg.weather["operational_start"]).date()
    if not refresh:
        cached = _hourly_cache_slice(
            cache,
            start=start,
            end=cfg.raw_end,
            timezone=cfg.weather["timezone"],
        )
        if cached is not None:
            return validate_weather_frame(
                cached,
                ["Temp_operational", "Wind_operational"],
            )
    session = session or resilient_session()
    variables = list(cfg.weather["operational_hourly_vars"])
    city_hourly: dict[str, pd.DataFrame] = {}
    for city in cfg.cities:
        parts = [
            _fetch_city_utc_hourly(
                cfg,
                city,
                session,
                variables,
                base_url=cfg.weather["previous_runs_url"],
                start=chunk_start,
                end=chunk_end,
            )
            for chunk_start, chunk_end in _date_chunks(start, cfg.raw_end)
        ]
        city_hourly[city.name] = pd.concat(parts).sort_index()
    out = combine_pop_weighted(city_hourly, cfg.city_weights(), _OPERATIONAL_RENAME)
    if out.isna().any().any():
        raise ValueError("operational hourly weather contains missing values")
    out = validate_weather_frame(
        out,
        ["Temp_operational", "Wind_operational"],
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache)
    return out
