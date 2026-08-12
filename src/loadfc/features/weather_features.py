"""Heating/cooling degree days and temperature-anomaly features."""

from __future__ import annotations

import numpy as np
import pandas as pd

_DAYS_PER_YEAR = 366


def _local_index(index: pd.Index) -> pd.DatetimeIndex:
    """Return the index as Berlin-local timestamps for calendar-day features."""

    local = pd.DatetimeIndex(index)
    if local.tz is not None:
        local = local.tz_convert("Europe/Berlin")
    return local


def add_degree_days(
    df: pd.DataFrame,
    hdd_threshold: float,
    cdd_threshold: float,
    temperature_col: str = "Temp_t",
) -> pd.DataFrame:
    out = df.copy()
    out["HDD"] = (hdd_threshold - out[temperature_col]).clip(lower=0.0)
    out["CDD"] = (out[temperature_col] - cdd_threshold).clip(lower=0.0)
    return out


def day_of_year_climatology(
    temperature: pd.Series,
    *,
    smooth_days: int = 7,
) -> pd.Series:
    """Return the smoothed mean temperature for each day of the year.

    The result is a fixed calendar transform: the same day of year always
    maps to the same climatological mean, so training and serving rows use
    identical values. Day-of-year never observed in the input gets the
    overall mean.
    """

    local = _local_index(temperature.index)
    daily = temperature.groupby(local.to_series().dt.month * 100 + local.day).mean()
    curve = pd.Series(np.nan, index=range(1, _DAYS_PER_YEAR + 1))
    for key, mean in daily.items():
        month, day = divmod(int(key), 100)
        curve.iloc[pd.Timestamp(2000, month, day).dayofyear - 1] = mean
    # Day-of-year never observed in the training window gets the training mean.
    curve = curve.fillna(float(daily.mean()))
    # Circular smoothing: pad head and tail, roll, trim.
    padded = pd.concat([curve.iloc[-smooth_days:], curve, curve.iloc[:smooth_days]])
    smoothed = padded.rolling(smooth_days * 2 + 1, center=True, min_periods=1).mean()
    return smoothed.iloc[smooth_days:-smooth_days]


def fit_temperature_climatology(
    temperature: pd.Series,
    validation_start,
    *,
    timezone: str = "Europe/Berlin",
) -> pd.Series:
    """Fit the calendar transform strictly before the local validation boundary."""

    index = pd.Index(temperature.index)
    if isinstance(index, pd.DatetimeIndex):
        berlin_midnight = pd.Timestamp(validation_start, tz=timezone)
        cutoff = (
            berlin_midnight.tz_convert(index.tz)
            if index.tz is not None
            else berlin_midnight.tz_localize(None)
        )
    else:
        cutoff = pd.Timestamp(validation_start).date()
    fitted = temperature.loc[index < cutoff]
    if fitted.empty:
        raise ValueError("temperature climatology requires observations before validation_start")
    return day_of_year_climatology(fitted)


def add_temperature_anomaly(
    df: pd.DataFrame,
    climatology: pd.Series,
    temperature_col: str = "Temp_forecast",
) -> pd.DataFrame:
    """Add the deviation of forecast temperature from its seasonal norm."""

    out = df.copy()
    doy = _local_index(out.index).dayofyear
    norm = pd.Series(doy, index=out.index).map(climatology)
    if norm.isna().any():
        raise ValueError("climatology does not cover the full index")
    out["Temp_anomaly"] = out[temperature_col] - norm
    return out
