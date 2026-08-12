"""Evaluation helpers for hourly multi-horizon forecasts."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd


def metrics_by_horizon(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return MAE and RMSE for each forecast step."""

    if "error" not in predictions:
        raise ValueError("hourly predictions require an error column")
    rows = []
    for horizon, group in predictions.groupby(level="horizon"):
        error = group["error"].to_numpy(dtype="float64")
        rows.append(
            {
                "horizon": int(horizon),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "n": int(error.size),
            }
        )
    return pd.DataFrame(rows).set_index("horizon").sort_index()


def aggregate_horizon_to_daily(
    predictions: pd.DataFrame,
    *,
    horizon: int = 24,
    timezone: str = "Europe/Berlin",
) -> pd.DataFrame:
    """Aggregate one fixed hourly horizon to German local daily means."""

    if not isinstance(predictions.index, pd.MultiIndex):
        raise ValueError("hourly predictions require a multi-index")
    required_levels = {"valid_time", "horizon"}
    if not required_levels <= set(predictions.index.names):
        raise ValueError("hourly prediction index requires valid_time and horizon")
    required_columns = {"actual", "prediction"}
    if not required_columns <= set(predictions):
        raise ValueError("hourly predictions require actual and prediction columns")
    selected = predictions[predictions.index.get_level_values("horizon") == horizon].copy()
    if selected.empty:
        raise ValueError(f"hourly predictions contain no horizon {horizon}")
    valid_time = pd.DatetimeIndex(selected.index.get_level_values("valid_time")).tz_convert(
        timezone
    )
    selected["date"] = valid_time.date
    return selected.groupby("date")[["actual", "prediction"]].mean()


def reconcile_to_daily_means(
    predictions: pd.DataFrame,
    daily_forecasts: pd.Series,
    *,
    horizon: int = 24,
    timezone: str = "Europe/Berlin",
) -> pd.DataFrame:
    """Scale a fixed-horizon hourly profile to coherent daily mean forecasts."""

    if not isinstance(predictions.index, pd.MultiIndex):
        raise ValueError("hourly predictions require a multi-index")
    required_levels = {"valid_time", "horizon"}
    if not required_levels <= set(predictions.index.names):
        raise ValueError("hourly prediction index requires valid_time and horizon")
    if "prediction" not in predictions:
        raise ValueError("hourly predictions require a prediction column")

    selected = predictions[predictions.index.get_level_values("horizon") == horizon].copy()
    if selected.empty:
        raise ValueError(f"hourly predictions contain no horizon {horizon}")

    anchors = daily_forecasts.copy()
    anchors.index = pd.Index(pd.to_datetime(anchors.index).date, name="date")
    if anchors.index.has_duplicates:
        raise ValueError("daily forecasts require one anchor per date")
    anchors = pd.to_numeric(anchors, errors="coerce")
    if anchors.isna().any() or not np.isfinite(anchors.to_numpy()).all():
        raise ValueError("daily forecasts must be finite")

    valid_time = pd.DatetimeIndex(selected.index.get_level_values("valid_time")).tz_convert(
        timezone
    )
    if valid_time.has_duplicates:
        raise ValueError("fixed-horizon predictions require unique valid times")
    local_dates = pd.Index(valid_time.date, name="date")
    missing_dates = sorted(set(local_dates) - set(anchors.index))
    if missing_dates:
        raise ValueError(f"daily forecasts are missing anchors for: {missing_dates}")

    observed_counts = pd.Series(1, index=local_dates).groupby(level="date").sum()
    expected_counts = pd.Series(
        {
            day: int(
                (
                    pd.Timestamp(day + timedelta(days=1), tz=timezone)
                    - pd.Timestamp(day, tz=timezone)
                )
                / pd.Timedelta(hours=1)
            )
            for day in observed_counts.index
        }
    )
    incomplete = observed_counts[observed_counts != expected_counts]
    if not incomplete.empty:
        raise ValueError(
            f"hourly predictions contain incomplete local days: {list(incomplete.index)}"
        )

    profile_means = (
        pd.Series(selected["prediction"].to_numpy(dtype="float64"), index=local_dates)
        .groupby(level="date")
        .mean()
    )
    if not np.isfinite(profile_means.to_numpy()).all() or (profile_means <= 0).any():
        raise ValueError("hourly profile means must be finite and positive")
    if (anchors.loc[profile_means.index] <= 0).any():
        raise ValueError("daily forecast anchors must be positive")

    selected["prediction_unreconciled"] = selected["prediction"]
    selected["daily_anchor"] = local_dates.map(anchors)
    factors = anchors.loc[profile_means.index] / profile_means
    selected["reconciliation_factor"] = local_dates.map(factors)
    selected["prediction"] = selected["prediction_unreconciled"] * selected["reconciliation_factor"]
    if "actual" in selected:
        selected["error"] = selected["actual"] - selected["prediction"]
    return selected


def reconciliation_invariant_rows(
    reconciled: pd.DataFrame,
    *,
    horizon: int = 24,
    timezone: str = "Europe/Berlin",
    tolerance: float = 1e-6,
) -> pd.DataFrame:
    """Report one anchor-coherence invariant for each complete local day."""

    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError("reconciliation tolerance must be finite and non-negative")
    if not isinstance(reconciled.index, pd.MultiIndex):
        raise ValueError("reconciled predictions require a multi-index")
    if {"valid_time", "horizon"} - set(reconciled.index.names):
        raise ValueError("reconciled predictions require valid_time and horizon")
    required = {"prediction", "daily_anchor"}
    if required - set(reconciled):
        raise ValueError(f"reconciled predictions require columns: {sorted(required)}")
    selected = reconciled[
        reconciled.index.get_level_values("horizon") == horizon
    ].copy()
    if selected.empty:
        raise ValueError(f"reconciled predictions contain no horizon {horizon}")
    valid_time = pd.DatetimeIndex(selected.index.get_level_values("valid_time"))
    if not valid_time.is_unique:
        raise ValueError("reconciled predictions require unique valid times")
    local_dates = pd.Index(valid_time.tz_convert(timezone).date, name="local_date")
    values = selected[["prediction", "daily_anchor"]].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError("reconciled invariant values must be finite")
    rows: list[dict[str, object]] = []
    for local_date, group in values.groupby(local_dates, sort=True):
        start = pd.Timestamp(local_date, tz=timezone)
        expected = len(
            pd.date_range(
                start,
                start + pd.DateOffset(days=1),
                freq="h",
                inclusive="left",
            )
        )
        if len(group) != expected:
            raise ValueError(f"reconciled predictions contain incomplete local days: {[local_date]}")
        anchor_values = group["daily_anchor"]
        if anchor_values.nunique() != 1:
            raise ValueError(f"reconciled daily anchor is not constant: {local_date}")
        mean = float(group["prediction"].mean())
        anchor = float(anchor_values.iat[0])
        delta = mean - anchor
        rows.append(
            {
                "local_date": local_date,
                "reconciled_hourly_mean": mean,
                "daily_anchor": anchor,
                "delta": delta,
                "abs_delta": abs(delta),
                "tolerance": float(tolerance),
                "pass": bool(abs(delta) <= tolerance),
                "n": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


reconciliation_invariants = reconciliation_invariant_rows
