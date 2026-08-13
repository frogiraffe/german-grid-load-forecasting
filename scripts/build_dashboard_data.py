"""Build the validated, static dashboard data contract from canonical results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# This stage reuses the validators of the preceding stage, so the repository root
# must be importable when the script runs as a subprocess rather than under pytest.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loadfc.config import Config  # noqa: E402
from loadfc.evaluation.protocol import (  # noqa: E402
    protocol_fingerprint,
    validate_protocol_record,
)
from loadfc.presentation import DAILY_PRESENTATION_MODEL  # noqa: E402
from loadfc.tracking import bundle_fingerprint, sha256_file, source_root  # noqa: E402
from scripts.validate_results import (  # noqa: E402
    _check_descriptive_study_evidence,
    _check_hourly_monthly_evidence,
    _validate_release_manifest,
)

_MAX_BYTES = 2 * 1024 * 1024
_DAILY_METRIC_COLUMNS = {
    "model",
    "evaluation_period",
    "stream_id",
    "protocol_fingerprint",
    "start",
    "end",
    "n",
    "RMSE",
    "MAE",
    "MAPE",
    "MASE",
}
_DAILY_PREDICTION_COLUMNS = {
    "date",
    "actual",
    "forecast",
    "stream_id",
    "protocol_fingerprint",
    "evaluation_period",
}
_DAILY_INTERVAL_COLUMNS = {
    "model",
    "method",
    "level",
    "evaluation_period",
    "stream_id",
    "protocol_fingerprint",
    "coverage_scope",
    "nominal",
    "empirical_coverage",
    "mean_width_MW",
    "interval_score_MW",
    "n",
}
_HOURLY_COMPARISON_COLUMNS = {
    "model",
    "slice_type",
    "slice_value",
    "evaluation_period",
    "stream_id",
    "protocol_fingerprint",
    "n",
    "RMSE",
    "MAE",
    "MAPE",
    "MASE",
}
_HOURLY_RAW_COLUMNS = {
    "prediction",
    "actual",
    "valid_time",
    "horizon",
    "stream_id",
    "protocol_fingerprint",
    "evaluation_period",
    "point_state_policy",
    "interval_state_policy",
    "weather_availability_assumption",
}
_HOURLY_RECONCILED_COLUMNS = {"prediction", "actual", "valid_time", "horizon"}
_HOURLY_INTERVAL_COLUMNS = {
    "method",
    "level",
    "slice_type",
    "slice_value",
    "evaluation_period",
    "coverage_scope",
    "stream_id",
    "protocol_fingerprint",
    "nominal",
    "empirical_coverage",
    "mean_width",
    "interval_score",
    "n",
}
_DAILY_MODELS = {
    "SARIMAX",
    "ensemble",
    "lightgbm",
    "naive_1d",
    "random_forest",
    "seasonal_naive_7d",
    "xgboost",
}


def _read_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"missing dashboard evidence: {path}")
    frame = pd.read_csv(path)
    if frame.empty or not required <= set(frame):
        raise ValueError(f"invalid dashboard evidence schema: {path}")
    return frame


def _finite(frame: pd.DataFrame, columns: tuple[str, ...], path: Path) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if not values.notna().all() or not values.map(math.isfinite).all():
            raise ValueError(f"non-finite {column}: {path}")


def _source_hashes(results: Path, summary: dict[str, object], paths: list[Path]) -> dict[str, str]:
    recorded = summary.get("artifact_hashes")
    if not isinstance(recorded, dict):
        raise ValueError("run summary artifact_hashes must be a mapping")
    hashes: dict[str, str] = {}
    for path in paths:
        relative = path.relative_to(results).as_posix()
        digest = sha256_file(path)
        if recorded.get(relative) != digest:
            raise ValueError(f"dashboard source hash conflict: {relative}")
        hashes[relative] = digest
    return dict(sorted(hashes.items()))


def _daily_payload(
    results: Path,
    summary: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], list[Path]]:
    comparison_path = results / "metrics/daily_comparison.csv"
    comparison = _read_csv(comparison_path, _DAILY_METRIC_COLUMNS)
    _finite(comparison, ("n", "RMSE", "MAE", "MAPE", "MASE"), comparison_path)
    if not comparison["evaluation_period"].eq("retrospective_final").all():
        raise ValueError(f"wrong daily evidence role: {comparison_path}")

    selected = DAILY_PRESENTATION_MODEL
    selected_metrics = comparison[comparison["model"].astype(str).eq(selected)]
    if len(selected_metrics) != 1:
        raise ValueError(f"daily selected model is missing or duplicated: {selected}")
    baselines = comparison[
        comparison["model"].astype(str).isin({"naive_1d", "seasonal_naive_7d"})
    ]
    if set(baselines["model"].astype(str)) != {"naive_1d", "seasonal_naive_7d"}:
        raise ValueError("daily baseline evidence is incomplete")

    prediction_path = results / f"predictions/{selected}.csv"
    interval_path = results / f"interval_predictions/{selected}.csv"
    coverage_path = results / "metrics/interval_coverage.csv"
    predictions = _read_csv(prediction_path, _DAILY_PREDICTION_COLUMNS)
    intervals = _read_csv(interval_path, _DAILY_PREDICTION_COLUMNS)
    coverage = _read_csv(coverage_path, _DAILY_INTERVAL_COLUMNS)
    _finite(predictions, ("actual", "forecast"), prediction_path)
    _finite(intervals, ("actual", "forecast"), interval_path)
    _finite(
        coverage,
        ("nominal", "empirical_coverage", "mean_width_MW", "interval_score_MW", "n"),
        coverage_path,
    )
    for frame, path in ((predictions, prediction_path), (intervals, interval_path)):
        if not frame["evaluation_period"].eq("retrospective_final").all():
            raise ValueError(f"wrong daily evidence role: {path}")
        if frame["stream_id"].nunique() != 1 or frame["protocol_fingerprint"].nunique() != 1:
            raise ValueError(f"mixed daily identity: {path}")
    selected_coverage = coverage[coverage["model"].astype(str).eq(selected)].copy()
    if selected_coverage.empty or not selected_coverage["evaluation_period"].eq(
        "retrospective_final"
    ).all():
        raise ValueError(f"daily interval evidence is incomplete: {selected}")

    identity = selected_metrics.iloc[0]
    expected_stream = str(identity["stream_id"])
    expected_fingerprint = str(identity["protocol_fingerprint"])
    for frame, path in (
        (predictions, prediction_path),
        (intervals, interval_path),
        (selected_coverage, coverage_path),
    ):
        if set(frame["stream_id"].astype(str)) != {expected_stream} or set(
            frame["protocol_fingerprint"].astype(str)
        ) != {expected_fingerprint}:
            raise ValueError(f"daily evidence identity conflict: {path}")

    if not predictions[["date", "actual", "forecast"]].equals(
        intervals[["date", "actual", "forecast"]]
    ):
        raise ValueError("daily point and interval rows disagree")
    bound_columns = sorted(
        column
        for column in intervals
        if column.startswith(("fixed_lower_", "fixed_upper_", "lower_", "upper_"))
    )
    if not bound_columns:
        raise ValueError(f"daily interval bounds are missing: {interval_path}")
    _finite(intervals, tuple(bound_columns), interval_path)
    metric = selected_metrics.iloc[0]
    forecast = {
        "model": selected,
        "start": str(metric["start"]),
        "end": str(metric["end"]),
        "n": int(metric["n"]),
        "rows": intervals[["date", "actual", "forecast", *bound_columns]].to_dict(
            orient="records"
        ),
    }
    interval_rows = selected_coverage.sort_values(["method", "level"]).to_dict(orient="records")
    overview = {
        "kind": "daily_comparison",
        "model": selected,
        "metrics": {
            key: float(metric[key]) for key in ("MAE", "RMSE", "MAPE", "MASE")
        },
        "baselines": baselines.sort_values("model")[
            ["model", "MAE", "RMSE", "MAPE", "MASE", "n"]
        ].to_dict(orient="records"),
        "n": int(metric["n"]),
    }
    paths = [comparison_path, prediction_path, interval_path, coverage_path]
    return forecast, interval_rows, [overview], paths


def _select_hourly_window(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    work = frame.copy()
    work["_utc"] = pd.to_datetime(work["valid_time"], utc=True, errors="coerce")
    if work["_utc"].isna().any() or work["_utc"].duplicated().any():
        raise ValueError("hourly valid_time values must be unique UTC timestamps")
    work["_local"] = work["_utc"].dt.tz_convert("Europe/Berlin")
    work["_local_date"] = work["_local"].dt.date

    complete: list[date] = []
    for local_day, rows in work.groupby("_local_date"):
        start = pd.Timestamp(local_day, tz="Europe/Berlin")
        end = pd.Timestamp(local_day + timedelta(days=1), tz="Europe/Berlin")
        expected = set(pd.date_range(start, end, inclusive="left", freq="h").tz_convert("UTC"))
        if set(rows["_utc"]) == expected:
            complete.append(local_day)

    windows = [
        complete[index - 6 : index + 1]
        for index in range(6, len(complete))
        if all(
            later - earlier == timedelta(days=1)
            for earlier, later in zip(
                complete[index - 6 : index],
                complete[index - 5 : index + 1],
                strict=True,
            )
        )
    ]
    if not windows:
        raise ValueError("hourly evidence has no complete seven-day Europe/Berlin window")
    chosen = windows[-1]
    selected = work[work["_local_date"].isin(chosen)].sort_values("_utc")
    return selected, {
        "local_start": chosen[0].isoformat(),
        "local_end": chosen[-1].isoformat(),
        "local_days": 7,
    }


def _hourly_forecast_payload(
    results: Path,
    models: list[str],
    selected_model: str,
    records: dict[str, dict[str, object]],
) -> tuple[dict[str, object], dict[str, object], list[Path]]:
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    paths: list[Path] = []
    assumptions: set[str] = set()
    point_policies: set[str] = set()
    interval_policies: set[str] = set()
    for model in models:
        stream_id = f"hourly/point/{model}"
        if stream_id not in records:
            raise ValueError(f"hourly model lacks protocol record: {model}")
        expected_fingerprint = protocol_fingerprint(records[stream_id])
        raw_path = results / f"hourly/test_predictions/{model}.csv"
        reconciled_path = results / f"hourly/test_reconciled_predictions/{model}.csv"
        raw = _read_csv(raw_path, _HOURLY_RAW_COLUMNS)
        reconciled = _read_csv(reconciled_path, _HOURLY_RECONCILED_COLUMNS)
        _finite(raw, ("prediction", "actual", "horizon"), raw_path)
        _finite(reconciled, ("prediction", "actual", "horizon"), reconciled_path)
        raw = raw[pd.to_numeric(raw["horizon"]).eq(24)].copy()
        reconciled = reconciled[pd.to_numeric(reconciled["horizon"]).eq(24)].copy()
        if raw.empty or reconciled.empty:
            raise ValueError(f"hourly h24 evidence is missing: {model}")
        if not raw["evaluation_period"].eq("retrospective_final").all():
            raise ValueError(f"wrong hourly prediction role: {raw_path}")
        if set(raw["stream_id"].astype(str)) != {stream_id} or set(
            raw["protocol_fingerprint"].astype(str)
        ) != {expected_fingerprint}:
            raise ValueError(f"hourly prediction identity conflict: {raw_path}")
        assumptions.update(raw["weather_availability_assumption"].astype(str))
        point_policies.update(raw["point_state_policy"].astype(str))
        interval_policies.update(raw["interval_state_policy"].astype(str))
        frames[(model, "raw")] = raw
        frames[(model, "reconciled")] = reconciled
        paths.extend((raw_path, reconciled_path))

    if len(assumptions) != 1 or len(point_policies) != 1 or len(interval_policies) != 1:
        raise ValueError("hourly prediction assumptions or state policies are mixed")
    selected, window = _select_hourly_window(frames[(selected_model, "raw")])
    expected_times = set(selected["_utc"])
    expected_actual = selected.set_index("_utc")["actual"].astype(float).to_dict()
    rows: list[dict[str, object]] = []
    for model in models:
        for state in ("raw", "reconciled"):
            frame = frames[(model, state)].copy()
            frame["_utc"] = pd.to_datetime(frame["valid_time"], utc=True, errors="coerce")
            if frame["_utc"].isna().any() or frame["_utc"].duplicated().any():
                raise ValueError(f"invalid hourly timestamps: {model}/{state}")
            window_rows = frame[frame["_utc"].isin(expected_times)].copy()
            if set(window_rows["_utc"]) != expected_times:
                raise ValueError(f"incomplete representative window: {model}/{state}")
            window_rows["_local"] = window_rows["_utc"].dt.tz_convert("Europe/Berlin")
            for _, row in window_rows.sort_values("_utc").iterrows():
                if float(row["actual"]) != expected_actual[row["_utc"]]:
                    raise ValueError(f"hourly actual values disagree: {model}/{state}")
                rows.append(
                    {
                        "model": model,
                        "state": state,
                        "valid_time": row["_utc"].isoformat(),
                        "local_time": row["_local"].isoformat(),
                        "actual": float(row["actual"]),
                        "forecast": float(row["prediction"]),
                        "horizon": int(row["horizon"]),
                    }
                )
    policies = {
        "weather_availability_assumption": next(iter(assumptions)),
        "point_state_policy": json.loads(next(iter(point_policies))),
        "interval_state_policy": json.loads(next(iter(interval_policies))),
    }
    return {"window": window, "rows": rows}, policies, paths


def _hourly_payload(
    results: Path,
    records: dict[str, dict[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]], list[Path]]:
    selection_path = results / "hourly/model_selection.csv"
    raw_path = results / "hourly/model_ablation_test.csv"
    reconciled_path = results / "hourly/model_ablation_test_reconciled.csv"
    hourly_interval_path = results / "hourly/interval_comparison.csv"
    monthly_interval_path = results / "hourly/coverage_by_month.csv"
    horizon_interval_path = results / "hourly/coverage_by_horizon.csv"
    comparison_path = results / "hourly/hourly_comparison.csv"
    horizon_path = results / "hourly/metrics_by_horizon.csv"
    reconciliation_path = results / "hourly/reconciliation_metrics.csv"
    invariant_path = results / "hourly/reconciliation_invariants.csv"
    bootstrap_path = results / "hourly/model_ablation_test_uncertainty.csv"
    selection = _read_csv(selection_path, {"selected_model", "selection_evidence_period"})
    raw = _read_csv(raw_path, {"model", "hourly_MAE", "n_hours"})
    reconciled = _read_csv(reconciled_path, {"model", "hourly_MAE", "n_hours"})
    hourly_interval = _read_csv(hourly_interval_path, _HOURLY_INTERVAL_COLUMNS)
    monthly_interval = _read_csv(monthly_interval_path, _HOURLY_INTERVAL_COLUMNS)
    horizon_interval = _read_csv(horizon_interval_path, _HOURLY_INTERVAL_COLUMNS)
    hourly_intervals = pd.concat(
        [hourly_interval, monthly_interval, horizon_interval], ignore_index=True
    )
    comparison = _read_csv(comparison_path, _HOURLY_COMPARISON_COLUMNS)
    horizons = _read_csv(horizon_path, {"model", "horizon", "mae", "rmse", "n"})
    reconciliation = _read_csv(
        reconciliation_path,
        {
            "model",
            "MAE",
            "RMSE",
            "MAPE",
            "n_hours",
            "evaluation_period",
            "stream_id",
            "protocol_fingerprint",
        },
    )
    invariants = _read_csv(
        invariant_path,
        {"local_date", "abs_delta", "tolerance", "pass", "n"},
    )
    bootstrap = _read_csv(
        bootstrap_path,
        {
            "candidate",
            "reference",
            "mae_difference",
            "ci_lower",
            "ci_upper",
            "probability_candidate_better",
            "n_days",
            "seed",
            "n_bootstrap",
            "block_size_days",
            "practical_tie_threshold",
            "practical_tie",
            "selected_model",
            "selected_model_preserved",
            "uncertainty_scope",
            "validation_selection_metric",
            "validation_selected_model",
            "validation_selection_evidence",
        },
    )
    for frame, columns, path in (
        (raw, ("hourly_MAE", "n_hours"), raw_path),
        (reconciled, ("hourly_MAE", "n_hours"), reconciled_path),
        (
            hourly_intervals,
            ("nominal", "empirical_coverage", "mean_width", "interval_score", "n"),
            hourly_interval_path,
        ),
        (comparison, ("n", "RMSE", "MAE", "MAPE", "MASE"), comparison_path),
        (horizons, ("horizon", "mae", "rmse", "n"), horizon_path),
        (reconciliation, ("MAE", "RMSE", "MAPE", "n_hours"), reconciliation_path),
        (invariants, ("abs_delta", "tolerance", "n"), invariant_path),
        (
            bootstrap,
            (
                "mae_difference",
                "ci_lower",
                "ci_upper",
                "probability_candidate_better",
                "n_days",
                "seed",
                "n_bootstrap",
                "block_size_days",
                "practical_tie_threshold",
            ),
            bootstrap_path,
        ),
    ):
        _finite(frame, columns, path)
    if len(selection) != 1 or selection["selection_evidence_period"].iat[0] != "validation":
        raise ValueError(f"invalid hourly selection evidence: {selection_path}")
    selected = str(selection["selected_model"].iat[0])
    raw_row = raw[raw["model"].astype(str).eq(selected)]
    reconciled_row = reconciled[reconciled["model"].astype(str).eq(selected)]
    if len(raw_row) != 1 or len(reconciled_row) != 1:
        raise ValueError(f"hourly selected model evidence is incomplete: {selected}")
    if not hourly_intervals["evaluation_period"].eq("retrospective_final").all():
        raise ValueError("wrong hourly interval role")
    if not hourly_interval["slice_type"].eq("aggregate").all():
        raise ValueError(f"wrong aggregate interval scope: {hourly_interval_path}")
    if not monthly_interval["slice_type"].eq("month").all():
        raise ValueError(f"wrong monthly interval scope: {monthly_interval_path}")
    if not horizon_interval["slice_type"].eq("horizon").all():
        raise ValueError(f"wrong horizon interval scope: {horizon_interval_path}")
    if not comparison["evaluation_period"].eq("retrospective_final").all():
        raise ValueError(f"wrong hourly comparison role: {comparison_path}")
    if not reconciliation["evaluation_period"].eq("retrospective_final").all():
        raise ValueError(f"wrong reconciliation role: {reconciliation_path}")
    if set(hourly_intervals["method"].astype(str)) != {"symmetric", "adaptive", "cqr"}:
        raise ValueError("hourly interval methods are incomplete")

    aggregate = comparison[comparison["slice_type"].eq("aggregate")]
    models = sorted(aggregate["model"].astype(str).unique())
    if not models or selected not in models or len(aggregate) != len(models):
        raise ValueError("hourly candidate aggregates are incomplete")
    for model in models:
        stream_id = f"hourly/point/{model}"
        record = records.get(stream_id)
        rows = comparison[comparison["model"].astype(str).eq(model)]
        if record is None or set(rows["stream_id"].astype(str)) != {stream_id} or set(
            rows["protocol_fingerprint"].astype(str)
        ) != {protocol_fingerprint(record)}:
            raise ValueError(f"hourly comparison identity conflict: {model}")
    for stream_id, rows in hourly_intervals.groupby("stream_id"):
        record = records.get(str(stream_id))
        if record is None or set(rows["protocol_fingerprint"].astype(str)) != {
            protocol_fingerprint(record)
        }:
            raise ValueError(f"hourly interval identity conflict: {stream_id}")

    hourly_forecast, policies, prediction_paths = _hourly_forecast_payload(
        results, models, selected, records
    )
    chosen_interval = hourly_interval.sort_values(["method"]).iloc[0]
    raw_mae = float(raw_row["hourly_MAE"].iat[0])
    reconciled_mae = float(reconciled_row["hourly_MAE"].iat[0])
    cards = [
        {
            "kind": "hourly_reconciliation",
            "model": selected,
            "raw_mae": raw_mae,
            "reconciled_mae": reconciled_mae,
            "mae_change": reconciled_mae - raw_mae,
            "n": int(reconciled_row["n_hours"].iat[0]),
        },
        {
            "kind": "interval_coverage",
            "method": str(chosen_interval["method"]),
            "nominal": float(chosen_interval["nominal"]),
            "empirical_coverage": float(chosen_interval["empirical_coverage"]),
            "n": int(chosen_interval["n"]),
        },
    ]
    error_rows = [
        {
            "model": str(row.model),
            "slice_type": str(row.slice_type),
            "slice_value": (
                int(float(row.slice_value))
                if str(row.slice_type) in {"horizon", "local_hour"}
                else str(row.slice_value)
            ),
            "n": int(row.n),
            "RMSE": float(row.RMSE),
            "MAE": float(row.MAE),
            "MAPE": float(row.MAPE),
            "MASE": float(row.MASE),
        }
        for row in comparison[
            comparison["slice_type"].isin({"aggregate", "horizon", "local_hour"})
        ].sort_values(["model", "slice_type", "slice_value"]).itertuples()
    ]
    horizon_rows = [
        {
            "model": str(row.model),
            "horizon": int(row.horizon),
            "MAE": float(row.mae),
            "RMSE": float(row.rmse),
            "n": int(row.n),
        }
        for row in horizons.sort_values(["model", "horizon"]).itertuples()
    ]
    interval_rows = [
        {
            "method": str(row.method),
            "level": str(row.level),
            "slice_type": str(row.slice_type),
            "slice_value": str(row.slice_value),
            "coverage_scope": str(row.coverage_scope),
            "nominal": float(row.nominal),
            "empirical_coverage": float(row.empirical_coverage),
            "mean_width": float(row.mean_width),
            "interval_score": float(row.interval_score),
            "n": int(row.n),
        }
        for row in hourly_intervals.sort_values(
            ["method", "slice_type", "slice_value"]
        ).itertuples()
    ]
    pass_values = invariants["pass"].astype(str).str.lower()
    if not pass_values.isin({"true", "false", "1", "0"}).all() or not pass_values.isin(
        {"true", "1"}
    ).all():
        raise ValueError("reconciliation invariant failed")
    complete_days = 0
    for row in invariants.itertuples():
        local_day = date.fromisoformat(str(row.local_date))
        start = pd.Timestamp(local_day, tz="Europe/Berlin")
        end = pd.Timestamp(local_day + timedelta(days=1), tz="Europe/Berlin")
        expected_n = len(pd.date_range(start, end, inclusive="left", freq="h"))
        complete_days += int(row.n) == expected_n
    raw_metric = reconciliation[~reconciliation["model"].astype(str).str.endswith("reconciled")]
    reconciled_metric = reconciliation[
        reconciliation["model"].astype(str).str.endswith("reconciled")
    ]
    if len(raw_metric) != 1 or len(reconciled_metric) != 1:
        raise ValueError("reconciliation metric states are incomplete")
    expected_stream = f"hourly/point/{selected}"
    expected_fingerprint = protocol_fingerprint(records[expected_stream])
    if set(reconciliation["stream_id"].astype(str)) != {expected_stream} or set(
        reconciliation["protocol_fingerprint"].astype(str)
    ) != {expected_fingerprint}:
        raise ValueError("reconciliation identity conflict")
    raw_value = float(raw_metric["MAE"].iat[0])
    reconciled_value = float(reconciled_metric["MAE"].iat[0])
    selection_record = {key: value.item() if hasattr(value, "item") else value for key, value in selection.iloc[0].items()}
    bootstrap_rows = [
        {key: value.item() if hasattr(value, "item") else value for key, value in row.items()}
        for row in bootstrap.to_dict(orient="records")
    ]
    payload = {
        "forecast": hourly_forecast,
        "intervals": interval_rows,
        "hourly_error": {
            "rows": error_rows,
            "horizon_profile": horizon_rows,
            "dst_sensitive_sample_sizes": len(set(invariants["n"].astype(int))) > 1
            or len(set(comparison[comparison["slice_type"].eq("local_hour")]["n"].astype(int)))
            > 1,
        },
        "reconciliation": {
            "model": selected,
            "raw_mae": raw_value,
            "reconciled_mae": reconciled_value,
            "absolute_change": reconciled_value - raw_value,
            "percentage_change": (reconciled_value - raw_value) / raw_value * 100.0,
            "n": int(reconciled_metric["n_hours"].iat[0]),
            "invariant": {
                "all_passed": True,
                "pass_count": len(invariants),
                "complete_local_days": complete_days,
                "max_abs_delta": float(invariants["abs_delta"].max()),
                "tolerance": float(invariants["tolerance"].max()),
            },
        },
        "selection": selection_record,
        "bootstrap": bootstrap_rows,
        "policies": policies,
        "models": models,
        "horizons": sorted({int(value) for value in horizons["horizon"]}),
        "interval_horizons": sorted(
            {
                int(float(value))
                for value in horizon_interval["slice_value"]
            }
        ),
        "local_hours": sorted(
            {
                int(float(value))
                for value in comparison.loc[
                    comparison["slice_type"].eq("local_hour"), "slice_value"
                ]
            }
        ),
    }
    paths = [
        selection_path,
        raw_path,
        reconciled_path,
        hourly_interval_path,
        monthly_interval_path,
        horizon_interval_path,
        comparison_path,
        horizon_path,
        reconciliation_path,
        invariant_path,
        bootstrap_path,
        *prediction_paths,
    ]
    return payload, cards, paths


def _study_payload(
    results: Path,
    records: dict[str, dict[str, object]],
) -> tuple[dict[str, object], list[Path]]:
    _check_descriptive_study_evidence(results)
    _check_hourly_monthly_evidence(results / "hourly", records)
    weekday_path = results / "analysis/load_profile_weekday.csv"
    month_path = results / "analysis/load_profile_month.csv"
    temperature_path = results / "analysis/temperature_load_curve.csv"
    shap_path = results / "analysis/shap_importance.csv"
    daily_path = results / "metrics/daily_comparison.csv"
    rolling_path = results / "metrics/rolling_origin_mape.csv"
    gap_path = results / "metrics/generalization_gap.csv"
    error_path = results / "metrics/error_slices.csv"
    feature_drift_path = results / "hourly/feature_drift.csv"
    residual_path = results / "hourly/residual_drift_monthly.csv"

    weekday = pd.read_csv(weekday_path)
    month = pd.read_csv(month_path)
    temperature = pd.read_csv(temperature_path)
    shap = _read_csv(shap_path, {"feature", "xgboost", "lightgbm", "mean"})
    _finite(shap, ("xgboost", "lightgbm", "mean"), shap_path)
    if len(shap) > 20 or not shap["feature"].astype(str).is_unique:
        raise ValueError(f"invalid bounded SHAP evidence: {shap_path}")

    daily = _read_csv(daily_path, _DAILY_METRIC_COLUMNS)
    _finite(daily, ("n", "RMSE", "MAE", "MAPE", "MASE"), daily_path)
    if len(daily) != 7 or set(daily["model"].astype(str)) != _DAILY_MODELS:
        raise ValueError(f"daily comparison is incomplete: {daily_path}")
    if not daily["evaluation_period"].eq("retrospective_final").all():
        raise ValueError(f"wrong daily comparison role: {daily_path}")
    for row in daily.itertuples():
        stream_id = str(row.stream_id)
        if stream_id not in records or row.protocol_fingerprint != protocol_fingerprint(
            records[stream_id]
        ):
            raise ValueError(f"daily comparison identity conflict: {daily_path}")

    rolling = pd.read_csv(rolling_path)
    if rolling.columns[0] not in {"period", "Unnamed: 0"}:
        raise ValueError(f"rolling-origin period column is invalid: {rolling_path}")
    rolling = rolling.rename(columns={rolling.columns[0]: "period"})
    rolling_models = [column for column in rolling if column != "period"]
    if (
        len(rolling) != 4
        or set(rolling_models) != _DAILY_MODELS - {"ensemble"}
        or not rolling["period"].astype(str).is_unique
    ):
        raise ValueError(f"rolling-origin evidence is incomplete: {rolling_path}")
    _finite(rolling, tuple(rolling_models), rolling_path)
    rolling_rows = rolling.melt(
        id_vars="period", var_name="model", value_name="MAPE"
    ).assign(evaluation_period="validation")

    gap = _read_csv(
        gap_path,
        {"model", "validation_MAPE", "test_MAPE", "absolute_gap_pp", "relative_gap_pct"},
    )
    _finite(
        gap,
        ("validation_MAPE", "test_MAPE", "absolute_gap_pp", "relative_gap_pct"),
        gap_path,
    )
    if len(gap) != 7 or set(gap["model"].astype(str)) != _DAILY_MODELS:
        raise ValueError(f"generalization-gap evidence is incomplete: {gap_path}")
    gap = gap.rename(columns={"test_MAPE": "retrospective_final_MAPE"}).assign(
        evaluation_period="retrospective_final"
    )

    errors = _read_csv(
        error_path,
        {"period", "model", "slice_type", "slice", "n", "MAE", "MAPE", "bias_MW"},
    )
    _finite(errors, ("n", "MAE", "MAPE", "bias_MW"), error_path)
    errors["evaluation_period"] = errors["period"].replace(
        {"test": "retrospective_final"}
    )
    expected_error_keys: set[tuple[str, str, str, str]] = set()
    for row in daily.itertuples():
        record = records[str(row.stream_id)]
        for role in ("validation", "retrospective_final"):
            start, end = record["splits"][role]
            expected_error_keys.update(
                (role, str(row.model), "month", month)
                for month in pd.period_range(start, end, freq="M").astype(str)
            )
            expected_error_keys.update(
                (role, str(row.model), "day_type", day_type)
                for day_type in ("weekday", "weekend")
            )
    actual_error_keys = set(
        zip(
            errors["evaluation_period"].astype(str),
            errors["model"].astype(str),
            errors["slice_type"].astype(str),
            errors["slice"].astype(str),
            strict=True,
        )
    )
    if (
        len(errors) != 210
        or set(errors["model"].astype(str)) != _DAILY_MODELS
        or actual_error_keys != expected_error_keys
        or errors.duplicated(["evaluation_period", "model", "slice_type", "slice"]).any()
    ):
        raise ValueError(f"error-slice evidence is incomplete: {error_path}")
    errors = errors.drop(columns="period")

    ablation_rows: list[dict[str, object]] = []
    ablation_paths: list[Path] = []
    for role, state, name in (
        ("validation", "raw", "model_ablation_validation.csv"),
        ("validation", "reconciled", "model_ablation_validation_reconciled.csv"),
        ("retrospective_final", "raw", "model_ablation_test.csv"),
        ("retrospective_final", "reconciled", "model_ablation_test_reconciled.csv"),
    ):
        path = results / "hourly" / name
        frame = _read_csv(path, {"model", "hourly_MAE", "hourly_RMSE", "hourly_MAPE", "n_hours"})
        _finite(frame, ("hourly_MAE", "hourly_RMSE", "hourly_MAPE", "n_hours"), path)
        if len(frame) != 3 or set(frame["model"].astype(str)) != {
            "residual_hybrid",
            "lightgbm_direct",
            "ridge_direct",
        }:
            raise ValueError(f"hourly ablation is incomplete: {path}")
        ablation_rows.extend(
            {**row, "evaluation_period": role, "state": state}
            for row in frame.to_dict(orient="records")
        )
        ablation_paths.append(path)

    feature_drift = _read_csv(feature_drift_path, {"feature", "psi", "status"})
    _finite(feature_drift, ("psi",), feature_drift_path)
    if len(feature_drift) != 6 or not feature_drift["feature"].astype(str).is_unique:
        raise ValueError(f"feature drift is incomplete: {feature_drift_path}")
    feature_rows = feature_drift.assign(
        evidence_scope="retrospective_explanation"
    ).to_dict(orient="records")
    residual = pd.read_csv(residual_path)

    return {
        "load_patterns": {
            "weekday": weekday.to_dict(orient="records"),
            "month": month.to_dict(orient="records"),
            "temperature": temperature.to_dict(orient="records"),
            "shap_importance": shap.assign(
                evidence_scope="retrospective_explanation"
            ).to_dict(orient="records"),
        },
        "performance": {
            "daily_comparison": daily.to_dict(orient="records"),
            "rolling_origin_mape": rolling_rows.to_dict(orient="records"),
            "generalization_gap": gap.to_dict(orient="records"),
            "hourly_ablation": ablation_rows,
        },
        "reliability": {
            "error_slices": errors.to_dict(orient="records"),
            "feature_drift": feature_rows,
            "residual_drift_monthly": residual.to_dict(orient="records"),
        },
        "method": {
            "evidence_roles": {
                "load_patterns": "full_available_history_descriptive",
                "selection": "validation",
                "feature_attribution": "retrospective_explanation",
                "final_results": "retrospective_final",
            }
        },
    }, [
        weekday_path,
        month_path,
        temperature_path,
        shap_path,
        daily_path,
        rolling_path,
        gap_path,
        error_path,
        *ablation_paths,
        feature_drift_path,
        residual_path,
    ]


def _build_release_payload(results: Path) -> dict[str, object]:
    summary_path = results / "run_summary.json"
    manifest_path = results / "evaluation_protocol.json"
    if not summary_path.is_file() or not manifest_path.is_file():
        raise ValueError("dashboard generation requires run summary and protocol manifest")
    summary = json.loads(summary_path.read_text())
    if summary.get("final_role") != "retrospective_final":
        raise ValueError("dashboard requires retrospective_final evidence")
    manifest = json.loads(manifest_path.read_text())
    records_payload = manifest.get("records") if isinstance(manifest, dict) else None
    if manifest.get("schema_version") != 1 or not isinstance(records_payload, dict):
        raise ValueError("invalid evaluation protocol manifest")
    records = {
        stream_id: validate_protocol_record(record)
        for stream_id, record in records_payload.items()
    }
    fingerprints = {
        stream_id: protocol_fingerprint(record) for stream_id, record in records.items()
    }
    if summary.get("protocol_fingerprints") != fingerprints:
        raise ValueError("dashboard protocol fingerprints conflict with run summary")
    for record in records.values():
        if record["source_revision"] != summary.get("source_revision") or record[
            "config_sha256"
        ] != summary.get("config_sha256"):
            raise ValueError("dashboard protocol identity conflicts with run summary")

    forecast, daily_intervals, overview, daily_paths = _daily_payload(results, summary)
    daily_methods = {str(row["method"]) for row in daily_intervals}
    if daily_methods != {"fixed", "adaptive"}:
        raise ValueError("daily interval methods are incomplete")
    hourly, hourly_overview, hourly_paths = _hourly_payload(results, records)
    study, study_paths = _study_payload(results, records)
    daily_stream = f"daily/{forecast['model']}"
    selected_hourly = str(hourly["selection"]["selected_model"])
    hourly_stream = f"hourly/point/{selected_hourly}"
    daily_record = records.get(daily_stream)
    hourly_record = records.get(hourly_stream)
    if daily_record is None or hourly_record is None:
        raise ValueError("selected dashboard models lack protocol records")
    if daily_record["splits"] != hourly_record["splits"]:
        raise ValueError("daily and hourly protocol periods disagree")
    adaptive_scope = next(
        (
            str(row["coverage_scope"])
            for row in hourly["intervals"]
            if row["method"] == "adaptive"
        ),
        None,
    )
    if adaptive_scope is None:
        raise ValueError("adaptive interval scope is missing")

    source_paths = [manifest_path, *daily_paths, *hourly_paths, *study_paths]
    source_paths = list(dict.fromkeys(source_paths))
    return {
        "schema_version": 1,
        "source_revision": summary["source_revision"],
        "bundle_fingerprint": summary["bundle_fingerprint"],
        "generated_at": summary["generated_at_utc"],
        "final_role": summary["final_role"],
        "source_artifact_hashes": _source_hashes(results, summary, source_paths),
        "selector_options": {
            "frequency": ["daily", "hourly"],
            "daily_models": [forecast["model"]],
            "hourly_models": hourly["models"],
            "hourly_horizons": hourly["horizons"],
            "hourly_point_error_models": sorted(
                {
                    str(row["model"])
                    for row in hourly["hourly_error"]["horizon_profile"]
                }
            ),
            "hourly_point_error_horizons": hourly["horizons"],
            "hourly_interval_coverage_horizons": hourly["interval_horizons"],
            "local_hours": hourly["local_hours"],
            "output_states": ["raw", "reconciled"],
            "daily_interval_methods": sorted({str(row["method"]) for row in daily_intervals}),
            "daily_interval_levels": sorted({str(row["level"]) for row in daily_intervals}),
            "hourly_interval_methods": sorted(
                {str(row["method"]) for row in hourly["intervals"]}
            ),
            "hourly_interval_levels": sorted(
                {str(row["level"]) for row in hourly["intervals"]}
            ),
        },
        "overview": [*overview, *hourly_overview],
        "forecast": {"daily": forecast, "hourly": hourly["forecast"]},
        "intervals": {"daily": daily_intervals, "hourly": hourly["intervals"]},
        "hourly_error": hourly["hourly_error"],
        "reconciliation": hourly["reconciliation"],
        "comparison": {
            "selection": hourly["selection"],
            "bootstrap": hourly["bootstrap"],
        },
        "study": study,
        "protocol": {
            "periods": daily_record["splits"],
            "weather_strategy": daily_record["weather_strategy"],
            "weather_availability_assumption": hourly["policies"][
                "weather_availability_assumption"
            ],
            "point_state_policy": {
                "daily": daily_record["point_state_policy"],
                "hourly": hourly_record["point_state_policy"],
            },
            "interval_state_policy": {
                "daily": daily_record["interval_state_policy"],
                "hourly": hourly_record["interval_state_policy"],
            },
            "selected_models": {
                "daily": forecast["model"],
                "hourly": selected_hourly,
                "daily_anchor": hourly["selection"].get("daily_anchor_model"),
            },
            "config_sha256": summary["config_sha256"],
            "lock_sha256": summary["lock_sha256"],
            "manifest_status": "hash_verified",
        },
        "limitations": [
            {"kind": "retrospective_final", "evidence": summary["final_role"]},
            {
                "kind": "hourly_forecast_definition",
                "evidence": "fixed 24-hour-ahead forecasts; hourly valid times do not share one common issue time",
            },
            {
                "kind": "weather_run_provenance",
                "evidence": hourly["policies"]["weather_availability_assumption"],
            },
            {"kind": "adaptive_interval_scope", "evidence": adaptive_scope},
        ],
    }


def _encoded(payload: dict[str, object]) -> bytes:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if len(encoded) > _MAX_BYTES:
        raise ValueError(f"dashboard payload exceeds {_MAX_BYTES} bytes")
    return encoded


def _write_release_and_manifest(
    destination: Path,
    summary_path: Path,
    payload: dict[str, object],
) -> None:
    release_bytes = _encoded(payload)
    summary = json.loads(summary_path.read_text())
    declared = summary.get("declared_artifacts")
    if not isinstance(declared, dict):
        raise ValueError("run summary declared_artifacts must be a mapping")
    declared["dashboard/src/generated/release.json"] = hashlib.sha256(release_bytes).hexdigest()
    fingerprint_fields = {
        key: summary[key]
        for key in (
            "source_revision",
            "source_clean",
            "config_sha256",
            "lock_sha256",
            "protocol_fingerprints",
            "declared_artifacts",
        )
    }
    summary["bundle_fingerprint"] = bundle_fingerprint(fingerprint_fields)
    if payload["bundle_fingerprint"] != summary["bundle_fingerprint"]:
        raise ValueError("dashboard bundle identity changed during declaration")
    summary_bytes = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode()

    destination.parent.mkdir(parents=True, exist_ok=True)
    release_tmp = destination.with_name(f".{destination.name}.tmp")
    summary_tmp = summary_path.with_name(f".{summary_path.name}.tmp")
    release_backup = destination.with_name(f".{destination.name}.backup")
    summary_backup = summary_path.with_name(f".{summary_path.name}.backup")
    if release_backup.exists() or summary_backup.exists():
        raise RuntimeError("stale dashboard write backup blocks replacement")
    release_tmp.write_bytes(release_bytes)
    summary_tmp.write_bytes(summary_bytes)
    release_backed_up = False
    summary_backed_up = False
    release_written = False
    summary_written = False
    try:
        if destination.exists():
            destination.replace(release_backup)
            release_backed_up = True
        summary_path.replace(summary_backup)
        summary_backed_up = True
        release_tmp.replace(destination)
        release_written = True
        summary_tmp.replace(summary_path)
        summary_written = True
    except BaseException:
        if summary_written:
            summary_path.replace(summary_tmp)
        if release_written:
            destination.replace(release_tmp)
        if release_backed_up and release_backup.exists():
            release_backup.replace(destination)
        if summary_backed_up and summary_backup.exists():
            summary_backup.replace(summary_path)
        raise
    if release_backup.exists():
        release_backup.unlink()
    summary_backup.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = Config.from_yaml(config_path)
    results = cfg.path("results_dir")
    manifest = json.loads((results / "evaluation_protocol.json").read_text())
    records_payload = manifest.get("records") if isinstance(manifest, dict) else None
    if manifest.get("schema_version") != 1 or not isinstance(records_payload, dict):
        raise ValueError("invalid evaluation protocol manifest")
    records = {
        stream_id: validate_protocol_record(record)
        for stream_id, record in records_payload.items()
    }
    summary_path = results / "run_summary.json"
    summary = json.loads(summary_path.read_text())
    _validate_release_manifest(summary, results, records, source_root(config_path), config_path)
    payload = _build_release_payload(results)
    destination = config_path.parent / "dashboard/src/generated/release.json"
    _write_release_and_manifest(destination, summary_path, payload)
    print("wrote", destination)


if __name__ == "__main__":
    main()
