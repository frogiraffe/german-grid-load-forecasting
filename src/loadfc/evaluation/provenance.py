"""Prediction-artifact identity, weather provenance, and output-root policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

_METADATA = ["weather_source_run", "weather_availability_assumption"]


def artifact_results_root(cfg: Any) -> Path:
    """Keep oracle sensitivity artifacts outside the canonical results root."""

    root = cfg.path("results_dir")
    if cfg.features.get("weather_strategy", "persistence") == "oracle":
        return root / "sensitivity" / "oracle"
    return root


def _weather_metadata(
    weather: pd.DataFrame,
    index: pd.Index,
    *,
    weather_strategy: str,
) -> pd.DataFrame:
    if weather_strategy == "oracle":
        return pd.DataFrame(
            {
                "weather_source_run": "oracle_sensitivity",
                "weather_availability_assumption": "realized weather is sensitivity-only",
            },
            index=index,
        )
    if weather_strategy == "persistence":
        return pd.DataFrame(
            {
                "weather_source_run": "persistence",
                "weather_availability_assumption": (
                    "previous day's realized observation treated as available at forecast origin"
                ),
            },
            index=index,
        )
    if weather_strategy != "available_day_ahead":
        raise ValueError(f"unknown weather strategy: {weather_strategy!r}")
    required = {"Temp_operational", "Wind_operational"}
    if not required <= set(weather):
        raise ValueError("available_day_ahead provenance requires operational weather columns")

    selected = weather.reindex(index)
    temp_available = selected["Temp_operational"].notna()
    wind_available = selected["Wind_operational"].notna()
    source = pd.Series("persistence", index=index, dtype="object")
    source.loc[temp_available & wind_available] = "open_meteo_previous_day1"
    source.loc[temp_available & ~wind_available] = "open_meteo_previous_day1_temperature_persistence_wind"
    source.loc[~temp_available & wind_available] = "persistence_temperature_open_meteo_previous_day1_wind"
    assumption = pd.Series(
        "previous day's realized observation treated as available at forecast origin",
        index=index,
        dtype="object",
    )
    assumption.loc[temp_available & wind_available] = (
        "previous_day1 treated as available 24 hours before valid_time; "
        "exact provider run timestamp unavailable"
    )
    assumption.loc[temp_available & ~wind_available] = (
        "temperature previous_day1 treated as available 24 hours before valid_time; "
        "wind uses previous day's realized observation; exact provider run timestamp unavailable"
    )
    assumption.loc[~temp_available & wind_available] = (
        "wind previous_day1 treated as available 24 hours before valid_time; "
        "temperature uses previous day's realized observation; exact provider run timestamp unavailable"
    )
    return pd.DataFrame(
        {"weather_source_run": source, "weather_availability_assumption": assumption},
        index=index,
    )


def _validated_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"actual", "forecast"}
    if predictions.empty:
        raise ValueError("prediction artifact must contain at least one row")
    if not required <= set(predictions):
        raise ValueError("prediction artifact requires actual and forecast columns")
    if isinstance(predictions.index, pd.MultiIndex):
        has_null_identity = any(
            pd.Index(predictions.index.get_level_values(level)).hasnans
            for level in predictions.index.names
        )
    else:
        has_null_identity = predictions.index.hasnans
    if has_null_identity:
        raise ValueError("prediction artifact identity cannot be null")
    return predictions[["actual", "forecast"]].copy()


def daily_prediction_artifact(
    predictions: pd.DataFrame,
    weather: pd.DataFrame,
    *,
    weather_strategy: str,
) -> pd.DataFrame:
    """Attach stable daily identity and weather provenance immediately before CSV output."""

    out = _validated_rows(predictions)
    valid_time = pd.DatetimeIndex(pd.to_datetime(out.index, errors="coerce"))
    if valid_time.isna().any():
        raise ValueError("daily prediction artifact identity must be a valid timestamp")
    out["forecast_origin"] = valid_time - pd.Timedelta(days=1)
    out["valid_time"] = valid_time
    if out.duplicated(["forecast_origin", "valid_time"]).any():
        raise ValueError("daily prediction artifact contains duplicate primary identities")
    metadata = _weather_metadata(weather, out.index, weather_strategy=weather_strategy)
    out[_METADATA] = metadata
    if out[["forecast_origin", "valid_time", *_METADATA]].isna().any().any():
        raise ValueError("daily prediction artifact has null identity or provenance")
    return out.sort_values(["forecast_origin", "valid_time"], kind="stable")


def hourly_prediction_artifact(
    predictions: pd.DataFrame,
    weather: pd.DataFrame,
    *,
    weather_strategy: str,
) -> pd.DataFrame:
    """Attach weather provenance to an hourly artifact with explicit UTC identity columns."""

    _validated_rows(predictions)
    required = {"forecast_origin", "valid_time", "horizon"}
    if not required <= set(predictions):
        raise ValueError("hourly prediction artifact requires forecast_origin, valid_time, and horizon")
    out = predictions.copy()
    if out[["forecast_origin", "valid_time", "horizon"]].isna().any().any():
        raise ValueError("hourly prediction artifact identity cannot be null")
    for column in ("forecast_origin", "valid_time"):
        out[column] = pd.to_datetime(out[column], errors="coerce", utc=True)
    horizon = pd.to_numeric(out["horizon"], errors="coerce")
    if out[["forecast_origin", "valid_time"]].isna().any().any() or (
        horizon.isna().any() or (horizon <= 0).any() or (horizon % 1 != 0).any()
    ):
        raise ValueError("hourly prediction artifact identity is invalid")
    out["horizon"] = horizon.astype(int)
    expected = pd.to_timedelta(out["horizon"], unit="h")
    if not (out["valid_time"] - out["forecast_origin"]).eq(expected).all():
        raise ValueError("hourly prediction artifact valid_time must match horizon")
    if out.duplicated(["forecast_origin", "valid_time", "horizon"]).any():
        raise ValueError("hourly prediction artifact contains duplicate primary identities")
    metadata = _weather_metadata(
        weather,
        pd.Index(out["valid_time"]),
        weather_strategy=weather_strategy,
    )
    out[_METADATA] = metadata.to_numpy()
    if out[_METADATA].isna().any().any():
        raise ValueError("hourly prediction artifact has null provenance")
    order = out.reset_index(drop=True).sort_values(
        ["forecast_origin", "valid_time", "horizon"], kind="stable"
    ).index
    return out.iloc[order]
