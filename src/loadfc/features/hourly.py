"""Leakage-safe features for UTC-indexed hourly load forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from . import calendar, lags, weather_features

_HOURS_PER_DAY = 24
_HOURS_PER_WEEK = 168
_HOURS_PER_YEAR = 24 * 365.25


def build_hourly_features(frame: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Build valid-time features using only information available day-ahead."""

    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("hourly features require a timezone-aware DatetimeIndex")
    required = {"hourly_load", "Temp_t", "Wind_t"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"hourly dataset is missing columns: {sorted(missing)}")

    out = pd.DataFrame(index=frame.index)
    out["hourly_load"] = frame["hourly_load"]
    strategy = cfg.features.get("weather_strategy", "persistence")
    if strategy == "available_day_ahead":
        operational = {"Temp_operational", "Wind_operational"}
        if not operational <= set(frame.columns):
            raise ValueError("available_day_ahead weather requires hourly operational columns")
        out["Temp_forecast"] = frame["Temp_operational"].fillna(
            frame["Temp_t"].shift(_HOURS_PER_DAY)
        )
        out["Wind_forecast"] = frame["Wind_operational"].fillna(
            frame["Wind_t"].shift(_HOURS_PER_DAY)
        )
    else:
        shift = _HOURS_PER_DAY if strategy == "persistence" else 0
        out["Temp_forecast"] = frame["Temp_t"].shift(shift)
        out["Wind_forecast"] = frame["Wind_t"].shift(shift)

    degree_days = weather_features.add_degree_days(
        out,
        cfg.features["hdd_threshold"],
        cfg.features["cdd_threshold"],
        temperature_col="Temp_forecast",
    )
    out[["HDD", "CDD"]] = degree_days[["HDD", "CDD"]]
    climatology = weather_features.fit_temperature_climatology(
        frame["Temp_t"],
        cfg.split.train_end + pd.Timedelta(days=1),
    )
    out = weather_features.add_temperature_anomaly(out, climatology)

    local = frame.index.tz_convert("Europe/Berlin")
    local_dates = pd.Index(local.date)
    calendar_frame = calendar.calendar_features(local_dates, cfg.features.get("structural_breaks"))
    calendar_frame.index = frame.index
    out = out.join(calendar_frame)
    out["hour"] = local.hour
    out["sin_hour"] = np.sin(2 * np.pi * out["hour"] / _HOURS_PER_DAY)
    out["cos_hour"] = np.cos(2 * np.pi * out["hour"] / _HOURS_PER_DAY)

    # Absolute UTC hours keep Fourier phase stable across training, validation,
    # and serving slices.
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    position = ((frame.index - epoch) / pd.Timedelta(hours=1)).to_numpy(dtype="float64")
    out["trend_hour"] = position
    for name, period in (
        ("week", _HOURS_PER_WEEK),
        ("year", _HOURS_PER_YEAR),
    ):
        out[f"sin_1_{name}"] = np.sin(2 * np.pi * position / period)
        out[f"cos_1_{name}"] = np.cos(2 * np.pi * position / period)

    hourly_lags = list(cfg.features.get("hourly_lags", [24, 168]))
    out = out.join(lags.add_lags(frame["hourly_load"], hourly_lags))
    return out


def hourly_feature_matrix(features: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose day-ahead features or target lags are unavailable."""

    predictors = [column for column in features if column != "hourly_load"]
    return features[["hourly_load", *predictors]].dropna(subset=predictors)
