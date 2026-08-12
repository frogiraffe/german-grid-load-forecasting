"""Compose all feature families into the modelling matrix."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from ..config import Config
from . import calendar, fourier, lags, weather_features

_BASE = ["daily_load", "Temp_forecast", "Wind_forecast"]
_DERIVED = [
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
]
_LAGS = ["L_t-1", "L_t-7"]

_EXOG = [
    "Weekend",
    "holiday_national",
    "holiday_religious",
    "bridge_day",
    "pre_holiday",
    "post_holiday",
    "is_covid",
    "is_energy_crisis",
    "Temp_forecast",
    "Wind_forecast",
    "HDD",
    "CDD",
    "Temp_anomaly",
    "sin_1_week",
    "cos_1_week",
    "sin_1_year",
    "cos_1_year",
]


def build_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    f = cfg.features
    feats = pd.DataFrame(index=df.index)
    feats["daily_load"] = df["daily_load"]
    strategy = f.get("weather_strategy", "persistence")
    if strategy == "available_day_ahead":
        required = {"Temp_operational", "Wind_operational"}
        if not required <= set(df.columns):
            raise ValueError(
                "available_day_ahead weather requires Temp_operational and Wind_operational"
            )
        feats["Temp_forecast"] = df["Temp_operational"].fillna(df["Temp_t"].shift(1))
        feats["Wind_forecast"] = df["Wind_operational"].fillna(df["Wind_t"].shift(1))
    else:
        shift = 1 if strategy == "persistence" else 0
        feats["Temp_forecast"] = df["Temp_t"].shift(shift)
        feats["Wind_forecast"] = df["Wind_t"].shift(shift)
    dd = weather_features.add_degree_days(
        feats,
        f["hdd_threshold"],
        f["cdd_threshold"],
        temperature_col="Temp_forecast",
    )
    feats[["HDD", "CDD"]] = dd[["HDD", "CDD"]]
    climatology = weather_features.fit_temperature_climatology(
        df["Temp_t"],
        cfg.split.train_end + timedelta(days=1),
    )
    feats = weather_features.add_temperature_anomaly(feats, climatology)
    feats = feats.join(calendar.calendar_features(df.index, f.get("structural_breaks")))
    for spec in f["fourier"]:
        feats = feats.join(
            fourier.fourier_terms(df.index, spec["period"], spec["name"], spec["harmonics"])
        )
    feats = feats.join(lags.add_lags(df["daily_load"], f["lags"]))
    return feats[_BASE + _DERIVED + _LAGS]


def exog_columns(model: str) -> list[str]:
    if model == "ml":
        return _EXOG + _LAGS
    if model == "sarimax":
        return list(_EXOG)
    raise ValueError(f"unknown model: {model!r}")


def feature_matrix(feats: pd.DataFrame, model: str) -> pd.DataFrame:
    cols = exog_columns(model)
    return feats[["daily_load"] + cols].dropna(subset=cols)
