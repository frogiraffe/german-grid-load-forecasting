"""Assemble the clean daily panel from load + weather series."""

from __future__ import annotations

from datetime import date

import pandas as pd

from ..config import Config
from . import smard
from . import weather as weather_mod
from .validation import validate_model_panel


def assemble(
    load: pd.Series,
    weather: pd.DataFrame,
    dataset_start: date,
    operational_weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    df = pd.concat([load.rename("daily_load"), weather], axis=1, join="inner")
    if operational_weather is not None:
        df = df.join(operational_weather, how="left")
    df = df.sort_index()
    df = df[df.index >= dataset_start]
    df = df.dropna(subset=["daily_load", "Temp_t", "Wind_t"])
    if df.empty:
        raise ValueError("no overlap between load and weather after trimming to dataset_start")
    df.index.name = "date"
    columns = ["daily_load", "Temp_t", "Wind_t"]
    if operational_weather is not None:
        columns += ["Temp_operational", "Wind_operational"]
    return validate_model_panel(
        df[columns],
        load_column="daily_load",
        operational=operational_weather is not None,
    )


def build(cfg: Config, refresh: bool = False) -> pd.DataFrame:
    load = smard.load_daily(cfg, refresh=refresh)
    weather = weather_mod.weather_daily(cfg, refresh=refresh)
    operational = None
    if cfg.features.get("weather_strategy") == "available_day_ahead":
        operational = weather_mod.operational_weather_daily(cfg, refresh=refresh)
    df = assemble(load, weather, cfg.dataset_start, operational)
    out_dir = cfg.path("processed_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "dataset.parquet")
    return df


def assemble_hourly(
    load: pd.Series,
    weather: pd.DataFrame,
    dataset_start: date,
    operational_weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join hourly inputs on UTC instants without changing the daily path."""

    df = pd.concat([load.rename("hourly_load"), weather], axis=1, join="inner")
    if operational_weather is not None:
        df = df.join(operational_weather, how="left")
    df = df.sort_index()
    local_dates = df.index.tz_convert("Europe/Berlin").date
    df = df[local_dates >= dataset_start]
    df = df.dropna(subset=["hourly_load", "Temp_t", "Wind_t"])
    if df.empty:
        raise ValueError("no hourly overlap between load and weather after dataset_start")
    df.index.name = "timestamp_utc"
    columns = ["hourly_load", "Temp_t", "Wind_t"]
    if operational_weather is not None:
        columns += ["Temp_operational", "Wind_operational"]
    return validate_model_panel(
        df[columns],
        load_column="hourly_load",
        operational=operational_weather is not None,
    )


def build_hourly(cfg: Config, refresh: bool = False) -> pd.DataFrame:
    """Build the optional hourly panel alongside the existing daily dataset."""

    load = smard.load_hourly(cfg, refresh=refresh)
    weather = weather_mod.weather_hourly(cfg, refresh=refresh)
    operational = None
    if cfg.features.get("weather_strategy") == "available_day_ahead":
        operational = weather_mod.operational_weather_hourly(cfg, refresh=refresh)
    frame = assemble_hourly(load, weather, cfg.dataset_start, operational)
    out_dir = cfg.path("processed_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_dir / "dataset_hourly.parquet")
    return frame
