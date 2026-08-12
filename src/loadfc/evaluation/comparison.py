"""Row-aligned, provenance-preserving comparison helpers."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from loadfc.evaluation.metrics import all_metrics

_REQUIRED = {"actual", "forecast", "evaluation_period", "stream_id", "protocol_fingerprint"}
_HOURLY_IDENTITY = ["forecast_origin", "valid_time", "horizon"]


def validate_daily_artifact(frame: pd.DataFrame, *, name: str = "prediction") -> pd.DataFrame:
    """Validate and normalize one persisted daily prediction artifact."""

    missing = _REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"{name} artifact missing required columns: {sorted(missing)}")
    out = frame.copy()
    if "date" in out:
        dates = pd.to_datetime(out["date"], errors="coerce")
    elif "valid_time" in out:
        dates = pd.to_datetime(out["valid_time"], errors="coerce")
    else:
        raise ValueError(f"{name} artifact requires date or valid_time")
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError(f"{name} artifact dates must be valid and unique")
    for column in ("actual", "forecast"):
        values = pd.to_numeric(out[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"{name} artifact {column} values must be finite")
        out[column] = values.astype("float64")
    if out["evaluation_period"].nunique() != 1:
        raise ValueError(f"{name} artifact must contain one evaluation period")
    if out["stream_id"].nunique() != 1 or out["protocol_fingerprint"].nunique() != 1:
        raise ValueError(f"{name} artifact metadata must be constant")
    out["date"] = dates.dt.date
    return out.sort_values("date", kind="stable").reset_index(drop=True)


def compare_daily_artifacts(
    artifacts: Mapping[str, pd.DataFrame],
    *,
    naive_mae: float,
    evaluation_period: str = "retrospective_final",
) -> pd.DataFrame:
    """Score streams on their own eligible rows and emit traceable metric rows."""

    if not artifacts:
        raise ValueError("at least one daily artifact is required")
    frames = {name: validate_daily_artifact(raw, name=name) for name, raw in artifacts.items()}
    for name, frame in frames.items():
        if frame["evaluation_period"].iat[0] != evaluation_period:
            raise ValueError(f"{name} artifact has unexpected evaluation period")
    common = set.intersection(*(set(frame["date"]) for frame in frames.values()))
    if not common:
        raise ValueError("daily artifacts have no common eligible dates")
    eligible = pd.Index(sorted(common), name="date")
    rows: list[dict[str, object]] = []
    for name, frame in frames.items():
        scored_frame = frame.set_index("date").loc[eligible]
        dates = pd.DatetimeIndex(eligible)
        scored = all_metrics(scored_frame["actual"], scored_frame["forecast"], naive_mae)
        rows.append(
            {
                "model": name,
                "evaluation_period": evaluation_period,
                "stream_id": str(frame["stream_id"].iat[0]),
                "protocol_fingerprint": str(frame["protocol_fingerprint"].iat[0]),
                "dates": f"{dates.min().date().isoformat()}:{dates.max().date().isoformat()}",
                "start": dates.min().date().isoformat(),
                "end": dates.max().date().isoformat(),
                "n": int(len(scored_frame)),
                **scored,
            }
        )
    return pd.DataFrame(rows)


def assert_compatible_daily_artifacts(artifacts: Mapping[str, pd.DataFrame]) -> None:
    """Fail closed when persisted streams cannot be compared as one period."""

    if not artifacts:
        raise ValueError("at least one daily artifact is required")
    periods: set[str] = set()
    for name, raw in artifacts.items():
        frame = validate_daily_artifact(raw, name=name)
        periods.add(str(frame["evaluation_period"].iat[0]))
    if len(periods) != 1:
        raise ValueError("daily artifacts have incompatible evaluation periods")


def validate_hourly_artifact(frame: pd.DataFrame, *, name: str = "prediction") -> pd.DataFrame:
    """Validate one UTC-identified hourly prediction stream."""

    required = {*_REQUIRED, *_HOURLY_IDENTITY}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} artifact missing required columns: {sorted(missing)}")
    out = frame.copy()
    for column in ("forecast_origin", "valid_time"):
        out[column] = pd.to_datetime(out[column], errors="coerce", utc=True)
    out["horizon"] = pd.to_numeric(out["horizon"], errors="coerce")
    if out[_HOURLY_IDENTITY].isna().any().any() or (out["horizon"] % 1 != 0).any():
        raise ValueError(f"{name} artifact has invalid UTC identity")
    out["horizon"] = out["horizon"].astype(int)
    if not (out["valid_time"] - out["forecast_origin"]).eq(
        pd.to_timedelta(out["horizon"], unit="h")
    ).all():
        raise ValueError(f"{name} artifact valid_time must match horizon")
    if out.duplicated(_HOURLY_IDENTITY).any():
        raise ValueError(f"{name} artifact UTC identities must be unique")
    for column in ("actual", "forecast"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
        if out[column].isna().any() or not np.isfinite(out[column].to_numpy()).all():
            raise ValueError(f"{name} artifact {column} values must be finite")
    if out["evaluation_period"].nunique() != 1:
        raise ValueError(f"{name} artifact must contain one evaluation period")
    if out["stream_id"].nunique() != 1 or out["protocol_fingerprint"].nunique() != 1:
        raise ValueError(f"{name} artifact metadata must be constant")
    return out.sort_values(_HOURLY_IDENTITY, kind="stable").reset_index(drop=True)


def compare_hourly_artifacts(
    artifacts: Mapping[str, pd.DataFrame],
    *,
    naive_mae: float,
    evaluation_period: str = "retrospective_final",
    timezone: str = "Europe/Berlin",
) -> pd.DataFrame:
    """Compare direct hourly streams on common UTC identities and useful local slices."""

    if not artifacts:
        raise ValueError("at least one hourly artifact is required")
    frames = {name: validate_hourly_artifact(raw, name=name) for name, raw in artifacts.items()}
    if any(frame["evaluation_period"].iat[0] != evaluation_period for frame in frames.values()):
        raise ValueError("hourly artifacts have unexpected evaluation period")
    common = set.intersection(*(set(map(tuple, frame[_HOURLY_IDENTITY].to_numpy())) for frame in frames.values()))
    if not common:
        raise ValueError("hourly artifacts have no common eligible UTC identities")
    key = pd.DataFrame(sorted(common), columns=_HOURLY_IDENTITY)
    rows: list[dict[str, object]] = []
    slices: list[tuple[str, object, pd.DataFrame]] = [("aggregate", "all", key)]
    slices.append(("horizon", 24, key[key["horizon"] == 24]))
    local = key.copy()
    local_time = pd.DatetimeIndex(local["valid_time"]).tz_convert(timezone)
    local["local_hour"] = local_time.hour
    local["local_day"] = local_time.date
    for hour, group in local.groupby("local_hour", sort=True):
        slices.append(("local_hour", int(hour), group[_HOURLY_IDENTITY]))
    for day, group in local.groupby("local_day", sort=True):
        slices.append(("local_day", day.isoformat(), group[_HOURLY_IDENTITY]))
    for name, frame in frames.items():
        indexed = frame.set_index(_HOURLY_IDENTITY)
        for slice_type, slice_value, eligible in slices:
            if eligible.empty:
                continue
            scored_frame = indexed.loc[pd.MultiIndex.from_frame(eligible)]
            dates = pd.DatetimeIndex(scored_frame.index.get_level_values("valid_time"))
            rows.append({
                "model": name,
                "slice_type": slice_type,
                "slice_value": slice_value,
                "evaluation_period": evaluation_period,
                "stream_id": str(frame["stream_id"].iat[0]),
                "protocol_fingerprint": str(frame["protocol_fingerprint"].iat[0]),
                "eligible_horizon": ",".join(map(str, sorted(scored_frame.index.get_level_values("horizon").unique()))),
                "local_start": dates.min().isoformat(),
                "local_end": dates.max().isoformat(),
                "n": len(scored_frame),
                **all_metrics(scored_frame["actual"], scored_frame["forecast"], naive_mae),
            })
    return pd.DataFrame(rows)


def paired_local_day_bootstrap(
    artifacts: Mapping[str, pd.DataFrame],
    *,
    reference: str,
    selected_model: str,
    seed: int,
    n_bootstrap: int = 10_000,
    block_size: int = 7,
    practical_tie_threshold: float = 50.0,
    timezone: str = "Europe/Berlin",
) -> pd.DataFrame:
    """Bootstrap paired daily MAE differences using contiguous local-day blocks."""

    if reference not in artifacts:
        raise ValueError(f"unknown bootstrap reference model: {reference}")
    errors: dict[str, pd.Series] = {}
    periods: set[str] = set()
    for name, raw in artifacts.items():
        frame = validate_hourly_artifact(raw, name=name)
        periods.add(str(frame["evaluation_period"].iat[0]))
        frame = frame[frame["horizon"] == 24].copy()
        local_day = pd.DatetimeIndex(frame["valid_time"]).tz_convert(timezone).date
        frame["local_day"] = local_day
        errors[name] = frame
    if len(periods) != 1 or periods != {"retrospective_final"}:
        raise ValueError("paired bootstrap requires one retrospective_final evaluation period")
    identity_sets = {
        name: set(map(tuple, frame[_HOURLY_IDENTITY].to_numpy())) for name, frame in errors.items()
    }
    common_identity = set.intersection(*identity_sets.values())
    if not common_identity:
        raise ValueError("paired bootstrap requires common UTC identities")
    for name, frame in errors.items():
        frame = frame[frame[_HOURLY_IDENTITY].apply(tuple, axis=1).isin(common_identity)]
        for day, group in frame.groupby("local_day"):
            expected = pd.date_range(
                pd.Timestamp(day, tz=timezone),
                pd.Timestamp(day, tz=timezone) + pd.DateOffset(days=1),
                freq="h",
                inclusive="left",
            ).tz_convert("UTC")
            if set(group["valid_time"]) != set(expected):
                raise ValueError("paired bootstrap requires complete, identical local-day row sets")
        errors[name] = frame.assign(error=(frame["actual"] - frame["forecast"]).abs()).groupby("local_day")["error"].mean()
    rows: list[dict[str, object]] = []
    ref = errors[reference]
    rng = np.random.default_rng(seed)
    for name, candidate in errors.items():
        if name == reference:
            continue
        common = ref.index.intersection(candidate.index).sort_values()
        if len(common) < 2:
            raise ValueError("paired bootstrap requires at least two common days")
        differences = (candidate.loc[common] - ref.loc[common]).to_numpy(dtype="float64")
        blocks = [np.arange(start, min(start + block_size, len(differences))) for start in range(0, len(differences), block_size)]
        samples = np.empty(n_bootstrap, dtype="float64")
        for index in range(n_bootstrap):
            chosen = rng.integers(0, len(blocks), size=len(blocks))
            sampled = np.concatenate([differences[blocks[i]] for i in chosen])[: len(differences)]
            samples[index] = sampled.mean()
        lower, upper = np.quantile(samples, [0.025, 0.975])
        tie = bool(lower <= practical_tie_threshold and upper >= -practical_tie_threshold)
        rows.append({
            "candidate": name,
            "reference": reference,
            "mae_difference": float(differences.mean()),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "probability_candidate_better": float(np.mean(samples < 0)),
            "n_days": len(differences),
            "seed": seed,
            "n_bootstrap": n_bootstrap,
            "block_size_days": block_size,
            "practical_tie_threshold": practical_tie_threshold,
            "practical_tie": tie,
            "selected_model": selected_model,
            "selected_model_preserved": bool(tie and selected_model == reference),
            "uncertainty_scope": "retrospective_final realized-weather time-series block bootstrap",
        })
    return pd.DataFrame(rows)
