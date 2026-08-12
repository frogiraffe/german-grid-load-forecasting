"""Run the cached 24-step hourly forecast, reconciliation, and drift evaluation."""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from loadfc.config import Config
from loadfc.data.build_dataset import build_hourly
from loadfc.evaluation import metrics
from loadfc.evaluation.conformal import (
    adaptive_conformal_interval,
    horizon_conformal_intervals,
    horizon_cqr_intervals,
    interval_evidence,
)
from loadfc.evaluation.drift import feature_drift_report, page_hinkley
from loadfc.evaluation.hourly import (
    aggregate_horizon_to_daily,
    metrics_by_horizon,
    reconcile_to_daily_means,
    reconciliation_invariant_rows,
)
from loadfc.evaluation.protocol import (
    assert_compatible_artifacts,
    merge_protocol_manifest,
    protocol_fingerprint,
)
from loadfc.evaluation.provenance import artifact_results_root, hourly_prediction_artifact
from loadfc.features.assemble import (
    build_features as build_daily_features,
)
from loadfc.features.assemble import (
    exog_columns as daily_exog_columns,
)
from loadfc.features.assemble import (
    feature_matrix as daily_feature_matrix,
)
from loadfc.features.horizon import direct_horizon_frame
from loadfc.features.hourly import build_hourly_features, hourly_feature_matrix
from loadfc.models.hourly import (
    HourlyDirectForecaster,
    HourlyHybridForecaster,
    make_hourly_direct_lightgbm,
    make_hourly_direct_ridge,
    make_hourly_hybrid,
    make_hourly_quantile_lightgbm,
)
from loadfc.models.ml import make_estimator
from loadfc.tracking import git_commit, sha256_file

HourlyModel = HourlyHybridForecaster | HourlyDirectForecaster
MIN_HOURLY_GROUP_N = 23
_EVIDENCE_COLUMNS = [
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
]


def _date_mask(values: pd.Series, start, end) -> pd.Series:
    local_dates = pd.DatetimeIndex(values).tz_convert("Europe/Berlin").date
    return pd.Series((local_dates >= start) & (local_dates <= end), index=values.index)


def _fit_models(
    cfg: Config,
    long_frame: pd.DataFrame,
    *,
    train_end,
) -> dict[str, HourlyModel]:
    train_mask = _date_mask(long_frame["valid_time"], cfg.dataset_start, train_end)
    candidates: dict[str, HourlyModel] = {
        "residual_hybrid": make_hourly_hybrid(cfg),
        "ridge_direct": make_hourly_direct_ridge(),
        "lightgbm_direct": make_hourly_direct_lightgbm(cfg),
    }
    return {name: candidate.fit(long_frame[train_mask]) for name, candidate in candidates.items()}


def _select_daily_anchor(cfg: Config) -> str:
    validation = pd.read_csv(
        cfg.path("results_dir") / "metrics" / "validation_metrics.csv",
        index_col=0,
    )
    candidates = ["random_forest", "lightgbm"]
    missing = set(candidates) - set(validation.index)
    if missing:
        raise ValueError(
            f"validation metrics are missing daily anchor candidates: {sorted(missing)}"
        )
    return str(validation.loc[candidates, "MAE"].idxmin())


def _fit_frozen_daily_anchor(
    cfg: Config,
    kind: str,
) -> tuple[Any, pd.DataFrame, tuple[str, ...]]:
    dataset = pd.read_parquet(cfg.path("processed_dir") / "dataset.parquet")
    features = build_daily_features(dataset, cfg)
    columns = tuple(daily_exog_columns("ml"))
    matrix = daily_feature_matrix(features, "ml")
    training = matrix[matrix.index < cfg.split.calibration_start]
    if training.empty:
        raise ValueError("daily anchor has no pre-calibration training rows")
    params = dict(cfg.models[kind])
    params.setdefault("random_state", cfg.seed)
    estimator = make_estimator(kind, params)
    estimator.fit(
        training[list(columns)].astype("float64"),
        training["daily_load"].to_numpy(dtype="float64"),
    )
    return estimator, matrix, columns


def _predict_daily_anchors(
    estimator: Any,
    matrix: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    start,
    end,
) -> pd.Series:
    period = matrix[(matrix.index >= start) & (matrix.index <= end)]
    if period.empty:
        raise ValueError(f"daily anchor has no rows from {start} through {end}")
    prediction = np.asarray(
        estimator.predict(period[list(columns)].astype("float64")),
        dtype="float64",
    )
    if prediction.shape != (len(period),) or not np.isfinite(prediction).all():
        raise ValueError("daily anchor predictions must be finite")
    return pd.Series(prediction, index=period.index, name="forecast")


def _predict_period(
    model: HourlyModel,
    long_frame: pd.DataFrame,
    *,
    start,
    end,
) -> pd.DataFrame:
    predict_mask = _date_mask(long_frame["valid_time"], start, end)
    period = long_frame[predict_mask]
    prediction = model.predict(period)
    prediction["actual"] = period["hourly_load"].to_numpy(dtype="float64")
    prediction["error"] = prediction["actual"] - prediction["prediction"]
    prediction["seasonal_naive"] = period["L_t-168"].to_numpy(dtype="float64")
    prediction["seasonal_naive_error"] = prediction["actual"] - prediction["seasonal_naive"]
    prediction["forecast_origin"] = period["forecast_origin"].to_numpy()
    prediction["valid_time"] = period["valid_time"].to_numpy()
    prediction["horizon"] = period["horizon"].to_numpy(dtype=int)
    return prediction


def _restore_interval_identity(intervals: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    """Reattach source UTC identity after a positional conformal calculation."""

    keys = ["forecast_origin", "valid_time", "horizon"]
    if not isinstance(source.index, pd.MultiIndex) or set(keys) - set(source.index.names):
        raise ValueError("interval source requires forecast-origin hourly identity")
    if len(intervals) != len(source):
        raise ValueError("interval rows do not match their source identity")
    out = intervals.copy()
    for key in keys:
        out[key] = source.index.get_level_values(key).to_numpy()
    return out.set_index(keys, drop=False).sort_index()


def _hourly_artifact(
    predictions: pd.DataFrame,
    weather: pd.DataFrame,
    *,
    weather_strategy: str,
) -> pd.DataFrame:
    """Attach output-only weather provenance without changing model frames."""

    if "prediction" not in predictions:
        raise ValueError("hourly artifact requires prediction values")
    artifact = hourly_prediction_artifact(
        predictions.rename(columns={"prediction": "forecast"}),
        weather,
        weather_strategy=weather_strategy,
    )
    return artifact.rename(columns={"forecast": "prediction"})


def _write_hourly_artifact(
    path: Path,
    predictions: pd.DataFrame,
    weather: pd.DataFrame,
    *,
    weather_strategy: str,
    metadata: dict[str, str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = _hourly_artifact(predictions, weather, weather_strategy=weather_strategy)
    if metadata:
        artifact = artifact.assign(**metadata)
    artifact.to_csv(path, index=False)


def _canonical_hourly_selection(cfg: Config) -> tuple[str, str]:
    """Load frozen operational choices for an oracle sensitivity run."""

    path = cfg.path("results_dir") / "hourly" / "model_selection.csv"
    if not path.exists():
        raise ValueError("oracle hourly sensitivity requires canonical model_selection.csv")
    selection = pd.read_csv(path)
    required = {"selected_model", "daily_anchor_model"}
    if len(selection) != 1 or not required <= set(selection):
        raise ValueError("canonical hourly model_selection.csv is invalid")
    selected_model = selection.iloc[0]["selected_model"]
    anchor_model = selection.iloc[0]["daily_anchor_model"]
    supported = {"residual_hybrid", "ridge_direct", "lightgbm_direct"}
    if selected_model not in supported or anchor_model not in cfg.models:
        raise ValueError("canonical hourly model_selection.csv names an unsupported model")
    return str(selected_model), str(anchor_model)


def _model_ablation(predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, frame in predictions.items():
        hourly = frame[frame["horizon"] == 24]
        daily = aggregate_horizon_to_daily(frame, horizon=24)
        rows.append(
            {
                "model": name,
                "hourly_MAE": metrics.mae(hourly["actual"], hourly["prediction"]),
                "hourly_RMSE": metrics.rmse(hourly["actual"], hourly["prediction"]),
                "hourly_MAPE": metrics.mape(hourly["actual"], hourly["prediction"]),
                "daily_MAE": metrics.mae(daily["actual"], daily["prediction"]),
                "daily_RMSE": metrics.rmse(daily["actual"], daily["prediction"]),
                "daily_MAPE": metrics.mape(daily["actual"], daily["prediction"]),
                "n_hours": len(hourly),
                "n_days": len(daily),
            }
        )
    return pd.DataFrame(rows).set_index("model").sort_values("hourly_MAE")


def _temporal_model_ablation(
    predictions: dict[str, pd.DataFrame],
    *,
    timezone: str = "Europe/Berlin",
) -> pd.DataFrame:
    """Report fixed-horizon errors by local calendar month."""

    rows = []
    for name, frame in predictions.items():
        hourly = frame[frame["horizon"] == 24].copy()
        valid_time = pd.DatetimeIndex(hourly.index.get_level_values("valid_time")).tz_convert(
            timezone
        )
        hourly["period"] = valid_time.strftime("%Y-%m")
        for period, group in hourly.groupby("period"):
            error = group["actual"] - group["prediction"]
            rows.append(
                {
                    "period": period,
                    "model": name,
                    "MAE": metrics.mae(group["actual"], group["prediction"]),
                    "RMSE": metrics.rmse(group["actual"], group["prediction"]),
                    "MAPE": metrics.mape(group["actual"], group["prediction"]),
                    "bias": float(error.mean()),
                    "n_hours": len(group),
                }
            )
    return pd.DataFrame(rows).set_index(["period", "model"]).sort_index()


def _paired_daily_mae_bootstrap(
    predictions: dict[str, pd.DataFrame],
    *,
    reference: str,
    seed: int,
    n_bootstrap: int = 10_000,
    timezone: str = "Europe/Berlin",
) -> pd.DataFrame:
    """Bootstrap paired daily MAE differences without treating hours as IID."""

    if reference not in predictions:
        raise ValueError(f"unknown bootstrap reference model: {reference}")
    daily_errors: dict[str, pd.Series] = {}
    for name, frame in predictions.items():
        hourly = frame[frame["horizon"] == 24]
        valid_time = pd.DatetimeIndex(hourly.index.get_level_values("valid_time")).tz_convert(
            timezone
        )
        local_dates = pd.Index(valid_time.date, name="date")
        daily_errors[name] = (
            pd.Series(
                np.abs(hourly["actual"] - hourly["prediction"]).to_numpy(),
                index=local_dates,
            )
            .groupby(level="date")
            .mean()
        )

    rng = np.random.default_rng(seed)
    rows = []
    reference_errors = daily_errors[reference]
    for name, candidate_errors in daily_errors.items():
        if name == reference:
            continue
        common = reference_errors.index.intersection(candidate_errors.index)
        differences = (candidate_errors.loc[common] - reference_errors.loc[common]).to_numpy(
            dtype="float64"
        )
        if differences.size < 2:
            raise ValueError("paired bootstrap requires at least two common days")
        samples = rng.choice(differences, size=(n_bootstrap, differences.size), replace=True)
        bootstrap_means = samples.mean(axis=1)
        rows.append(
            {
                "candidate": name,
                "reference": reference,
                "mae_difference": float(differences.mean()),
                "ci_lower": float(np.quantile(bootstrap_means, 0.025)),
                "ci_upper": float(np.quantile(bootstrap_means, 0.975)),
                "probability_candidate_better": float(np.mean(bootstrap_means < 0)),
                "n_days": differences.size,
            }
        )
    return pd.DataFrame(rows).set_index(["candidate", "reference"])


def _reconciled_quantile_frame(
    reconciled_point: pd.DataFrame,
    lower_predictions: pd.DataFrame,
    upper_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Apply point-reconciliation factors to matching quantile predictions."""

    lower = lower_predictions.reindex(reconciled_point.index)["prediction"]
    upper = upper_predictions.reindex(reconciled_point.index)["prediction"]
    if lower.isna().any() or upper.isna().any():
        raise ValueError("quantile predictions do not align with reconciled points")
    factor = reconciled_point["reconciliation_factor"]
    scaled_lower = lower * factor
    scaled_upper = upper * factor
    if (scaled_lower > scaled_upper).any():
        raise ValueError("quantile predictions cross after reconciliation")
    return pd.DataFrame(
        {
            "actual": reconciled_point["actual"],
            "prediction": reconciled_point["prediction"],
            "horizon": reconciled_point["horizon"],
            "lower_quantile": scaled_lower,
            "upper_quantile": scaled_upper,
            "quantile_crossed": False,
        },
        index=reconciled_point.index,
    )


def _evidence_scope(method: str) -> str:
    return (
        "prequential_monitoring_no_unconditional_time_series_coverage"
        if method == "adaptive"
        else "empirical_retrospective"
    )


def _evidence_row(
    frame: pd.DataFrame,
    *,
    method: str,
    slice_type: str,
    slice_value: str | int,
    metadata: dict[str, str],
    alpha: float,
) -> dict[str, object]:
    evidence = interval_evidence(frame["actual"], frame["lower"], frame["upper"], alpha)
    return {
        "method": method,
        "level": f"{int((1.0 - alpha) * 100)}%",
        "slice_type": slice_type,
        "slice_value": slice_value,
        "evaluation_period": "retrospective_final",
        "coverage_scope": _evidence_scope(method),
        "stream_id": metadata["stream_id"],
        "protocol_fingerprint": metadata["protocol_fingerprint"],
        **evidence,
    }


def _complete_local_day(frame: pd.DataFrame) -> bool:
    valid_time = pd.DatetimeIndex(pd.to_datetime(frame["valid_time"], utc=True))
    if not valid_time.is_unique:
        return False
    local_day = valid_time.tz_convert("Europe/Berlin")[0].date()
    start = pd.Timestamp(local_day, tz="Europe/Berlin")
    expected = pd.date_range(start, start + pd.DateOffset(days=1), freq="h", inclusive="left").tz_convert(
        "UTC"
    )
    return len(frame) in {23, 24, 25} and valid_time.equals(expected)


def _interval_evidence_rows(
    intervals_by_method: dict[str, pd.DataFrame],
    *,
    metadata: dict[str, str],
    alpha: float,
) -> pd.DataFrame:
    """Emit aggregate, eligible-horizon, and complete Berlin-local-day evidence."""

    rows: list[dict[str, object]] = []
    for method, intervals in intervals_by_method.items():
        required = {"actual", "lower", "upper", "horizon", "valid_time"}
        if not required <= set(intervals):
            raise ValueError(f"{method} intervals are missing required evidence columns")
        rows.append(
            _evidence_row(
                intervals,
                method=method,
                slice_type="aggregate",
                slice_value="all",
                metadata=metadata,
                alpha=alpha,
            )
        )
        for horizon, group in intervals.groupby(intervals["horizon"], sort=True):
            if len(group) >= MIN_HOURLY_GROUP_N:
                rows.append(
                    _evidence_row(
                        group,
                        method=method,
                        slice_type="horizon",
                        slice_value=int(horizon),
                        metadata=metadata,
                        alpha=alpha,
                    )
                )
        local_dates = pd.DatetimeIndex(pd.to_datetime(intervals["valid_time"], utc=True)).tz_convert(
            "Europe/Berlin"
        ).date
        for local_day, group in intervals.assign(_local_day=local_dates).groupby("_local_day", sort=True):
            group = group.drop(columns="_local_day")
            if len(group) >= MIN_HOURLY_GROUP_N and _complete_local_day(group):
                rows.append(
                    _evidence_row(
                        group,
                        method=method,
                        slice_type="local_day",
                        slice_value=local_day.isoformat(),
                        metadata=metadata,
                        alpha=alpha,
                    )
                )
    return pd.DataFrame(rows, columns=_EVIDENCE_COLUMNS)


def _monthly_interval_evidence_rows(
    intervals_by_method: dict[str, pd.DataFrame],
    *,
    metadata: dict[str, str],
    alpha: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, intervals in intervals_by_method.items():
        required = {"actual", "lower", "upper", "valid_time"}
        if not required <= set(intervals):
            raise ValueError(f"{method} intervals are missing required evidence columns")
        months = pd.DatetimeIndex(pd.to_datetime(intervals["valid_time"], utc=True)).tz_convert(
            "Europe/Berlin"
        ).strftime("%Y-%m")
        for month, group in intervals.assign(_month=months).groupby("_month", sort=True):
            rows.append(
                _evidence_row(
                    group.drop(columns="_month"),
                    method=method,
                    slice_type="month",
                    slice_value=str(month),
                    metadata=metadata,
                    alpha=alpha,
                )
            )
    return pd.DataFrame(rows, columns=_EVIDENCE_COLUMNS)


def _monthly_residual_drift(
    test: pd.DataFrame,
    monitor: pd.DataFrame,
    *,
    metadata: dict[str, str],
) -> pd.DataFrame:
    if len(test) != len(monitor):
        raise ValueError("residual monitor rows do not match test valid_time rows")
    required_test = {"valid_time", "error"}
    required_monitor = {"statistic", "alert"}
    if not required_test <= set(test) or not required_monitor <= set(monitor):
        raise ValueError("monthly residual drift inputs are incomplete")
    valid_time = pd.to_datetime(test["valid_time"], utc=True, errors="coerce").reset_index(
        drop=True
    )
    error = pd.to_numeric(test["error"], errors="coerce").abs().reset_index(drop=True)
    statistic = pd.to_numeric(monitor["statistic"], errors="coerce").reset_index(drop=True)
    if valid_time.isna().any() or not np.isfinite(error).all() or not np.isfinite(statistic).all():
        raise ValueError("monthly residual drift inputs must be finite")
    frame = pd.DataFrame(
        {
            "month": valid_time.dt.tz_convert("Europe/Berlin").dt.strftime("%Y-%m"),
            "absolute_error": error,
            "statistic": statistic,
            "alert": monitor["alert"].astype(bool).reset_index(drop=True),
        }
    )
    grouped = frame.groupby("month", sort=True)
    report = grouped.agg(
        mean_absolute_error_MW=("absolute_error", "mean"),
        max_page_hinkley_statistic=("statistic", "max"),
        alert_count=("alert", "sum"),
        n=("alert", "size"),
    ).reset_index()
    report.insert(1, "evaluation_period", "retrospective_final")
    report.insert(2, "monitoring_scope", "prequential_monitoring")
    report.insert(3, "stream_id", metadata["stream_id"])
    report.insert(4, "protocol_fingerprint", metadata["protocol_fingerprint"])
    report["alert_count"] = report["alert_count"].astype(int)
    report["n"] = report["n"].astype(int)
    return report


def _local_hour_evidence_rows(
    intervals: pd.DataFrame,
    *,
    metadata: dict[str, str],
    alpha: float,
) -> pd.DataFrame:
    local_hours = pd.DatetimeIndex(pd.to_datetime(intervals["valid_time"], utc=True)).tz_convert(
        "Europe/Berlin"
    ).hour
    rows = [
        _evidence_row(
            group.drop(columns="_local_hour"),
            method="cqr",
            slice_type="local_hour",
            slice_value=int(hour),
            metadata=metadata,
            alpha=alpha,
        )
        for hour, group in intervals.assign(_local_hour=local_hours).groupby("_local_hour", sort=True)
        if len(group) >= MIN_HOURLY_GROUP_N
    ]
    return pd.DataFrame(rows, columns=_EVIDENCE_COLUMNS)


def _interval_comparison(
    symmetric: pd.DataFrame,
    cqr: pd.DataFrame,
    adaptive: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Retain the compact comparison helper used by existing callers/tests."""

    frames = [("symmetric", symmetric), ("cqr", cqr)]
    if adaptive is not None:
        frames.append(("adaptive", adaptive))
    return pd.DataFrame(
        {
            "method": name,
            "coverage": ((frame["actual"] >= frame["lower"]) & (frame["actual"] <= frame["upper"])).mean(),
            "mean_width": (frame["upper"] - frame["lower"]).mean(),
            "n": len(frame),
        }
        for name, frame in frames
    ).set_index("method")


def _coverage_by_local_hour(intervals: pd.DataFrame) -> pd.DataFrame:
    """Retain the legacy local-hour summary alongside protocol evidence."""

    valid_time = pd.DatetimeIndex(intervals.index.get_level_values("valid_time")).tz_convert(
        "Europe/Berlin"
    )
    frame = intervals.assign(
        local_hour=valid_time.hour,
        covered=(intervals["actual"] >= intervals["lower"])
        & (intervals["actual"] <= intervals["upper"]),
        width=intervals["upper"] - intervals["lower"],
    )
    return frame.groupby("local_hour").agg(
        coverage=("covered", "mean"),
        mean_width=("width", "mean"),
        n=("covered", "size"),
    )


def _protocol_metadata(record: dict[str, object], period: str) -> dict[str, str]:
    return {
        "stream_id": str(record["stream_id"]),
        "protocol_fingerprint": protocol_fingerprint(record),
        "evaluation_period": period,
        "point_state_policy": json.dumps(record["point_state_policy"], sort_keys=True),
        "interval_state_policy": json.dumps(record["interval_state_policy"], sort_keys=True),
    }


def _hourly_protocol_records(
    cfg: Config,
    *,
    selected_model: str,
    point_model: HourlyModel,
    daily_anchor_model: str,
    daily_columns: tuple[str, ...],
    quantile_models: dict[str, HourlyDirectForecaster],
    config_path: Path,
) -> dict[str, dict[str, object]]:
    point_columns = point_model.feature_columns
    quantile_columns = {name: model.feature_columns for name, model in quantile_models.items()}
    if point_columns is None or any(columns is None for columns in quantile_columns.values()):
        raise ValueError("hourly protocol requires fitted feature schemas")
    if len(set(quantile_columns.values())) != 1:
        raise ValueError("quantile models must share one ordered feature schema")
    splits = {
        "train": [cfg.dataset_start.isoformat(), cfg.split.train_end.isoformat()],
        "validation": [(cfg.split.train_end + timedelta(days=1)).isoformat(), cfg.split.val_end.isoformat()],
        "calibration": [cfg.split.calibration_start.isoformat(), cfg.split.calibration_end.isoformat()],
        "retrospective_final": [cfg.split.test_start.isoformat(), cfg.split.test_end.isoformat()],
    }
    point_policy = {"fit_through": cfg.split.val_end.isoformat(), "update": "none"}
    interval_policy = {
        "fixed_cqr": "calibration_scores_frozen",
        "adaptive": "updates_after_actual_only",
    }
    common: dict[str, object] = {
        "schema_version": 1,
        "source_revision": git_commit(cfg.root),
        "config_sha256": sha256_file(config_path),
        "seed": cfg.seed,
        "splits": splits,
        "weather_strategy": cfg.features.get("weather_strategy", "persistence"),
        "point_state_policy": point_policy,
        "interval_state_policy": interval_policy,
        "final_role": "retrospective_final",
        "validation_selection_evidence": "results/hourly/model_ablation_validation_reconciled.csv",
        "rationale": "Validation selects hourly and anchor models; calibration and final do not reselect.",
    }
    point_id = f"hourly/point/{selected_model}"
    anchor_id = f"hourly/anchor/{daily_anchor_model}"
    quantile_id = "hourly/quantile/lightgbm_cqr"
    return {
        point_id: {
            **common,
            "stream_id": point_id,
            "model_identity": f"reconciled_hourly:{selected_model}",
            "ordered_feature_columns": list(point_columns),
            "model_parameters": {"selected_model": selected_model, "horizon": 24},
        },
        anchor_id: {
            **common,
            "stream_id": anchor_id,
            "model_identity": f"daily_anchor:{daily_anchor_model}",
            "ordered_feature_columns": list(daily_columns),
            "model_parameters": dict(cfg.models[daily_anchor_model]),
        },
        quantile_id: {
            **common,
            "stream_id": quantile_id,
            "model_identity": "lightgbm_quantile_cqr",
            "ordered_feature_columns": list(next(iter(quantile_columns.values())) or ()),
            "model_parameters": {
                **dict(cfg.models["lightgbm"]),
                "objective": "quantile",
                "quantiles": [0.05, 0.95],
                "horizon": 24,
            },
        },
    }


def _daily_model_comparison(
    cfg: Config,
    hourly_test: pd.DataFrame,
    reconciled_test: pd.DataFrame,
    *,
    selected_model: str,
    anchor_model: str,
) -> pd.DataFrame:
    hourly_daily = aggregate_horizon_to_daily(hourly_test, horizon=24)
    reconciled_daily = aggregate_horizon_to_daily(reconciled_test, horizon=24)
    rows = [
        {
            "model": f"hourly_{selected_model}_h24_daily_mean",
            "MAE": metrics.mae(hourly_daily["actual"], hourly_daily["prediction"]),
            "RMSE": metrics.rmse(hourly_daily["actual"], hourly_daily["prediction"]),
            "MAPE": metrics.mape(hourly_daily["actual"], hourly_daily["prediction"]),
            "n_days": len(hourly_daily),
        },
        {
            "model": f"hourly_{selected_model}_h24_{anchor_model}_reconciled",
            "MAE": metrics.mae(reconciled_daily["actual"], reconciled_daily["prediction"]),
            "RMSE": metrics.rmse(reconciled_daily["actual"], reconciled_daily["prediction"]),
            "MAPE": metrics.mape(reconciled_daily["actual"], reconciled_daily["prediction"]),
            "n_days": len(reconciled_daily),
        },
    ]
    predictions_dir = cfg.path("results_dir") / "predictions"
    for path in sorted(predictions_dir.glob("*.csv")):
        frame = pd.read_csv(path)
        if not {"actual", "forecast"} <= set(frame):
            continue
        rows.append(
            {
                "model": path.stem,
                "MAE": metrics.mae(frame["actual"], frame["forecast"]),
                "RMSE": metrics.rmse(frame["actual"], frame["forecast"]),
                "MAPE": metrics.mape(frame["actual"], frame["forecast"]),
                "n_days": len(frame),
            }
        )
    return pd.DataFrame(rows).set_index("model").sort_values("MAE")


def _reconciliation_comparison(
    hourly_test: pd.DataFrame,
    reconciled_test: pd.DataFrame,
    *,
    selected_model: str,
    anchor_model: str,
    metadata: dict[str, str] | None = None,
) -> pd.DataFrame:
    rows = []
    for name, frame in (
        (f"hourly_{selected_model}_h24", hourly_test[hourly_test["horizon"] == 24]),
        (f"hourly_{selected_model}_h24_{anchor_model}_reconciled", reconciled_test),
    ):
        rows.append(
            {
                "model": name,
                "MAE": metrics.mae(frame["actual"], frame["prediction"]),
                "RMSE": metrics.rmse(frame["actual"], frame["prediction"]),
                "MAPE": metrics.mape(frame["actual"], frame["prediction"]),
                "n_hours": len(frame),
                **(metadata or {}),
            }
        )
    return pd.DataFrame(rows).set_index("model")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--build",
        action="store_true",
        help="fetch/build the hourly dataset instead of requiring its cache",
    )
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    cfg = Config.from_yaml(args.config)
    weather_strategy = cfg.features.get("weather_strategy", "persistence")
    frozen_selection = _canonical_hourly_selection(cfg) if weather_strategy == "oracle" else None

    dataset_path = cfg.path("processed_dir") / "dataset_hourly.parquet"
    if args.build:
        dataset = build_hourly(cfg, refresh=args.refresh)
    elif dataset_path.exists():
        dataset = pd.read_parquet(dataset_path)
    else:
        raise FileNotFoundError(f"{dataset_path} is missing; run with --build to create it")

    features = hourly_feature_matrix(build_hourly_features(dataset, cfg))
    long_frame = direct_horizon_frame(features, horizon=24)
    anchor_model = frozen_selection[1] if frozen_selection else _select_daily_anchor(cfg)
    validation_models = _fit_models(cfg, long_frame, train_end=cfg.split.train_end)
    validation_predictions = {
        name: _predict_period(
            candidate,
            long_frame,
            start=cfg.split.train_end + timedelta(days=1),
            end=cfg.split.val_end,
        )
        for name, candidate in validation_models.items()
    }
    validation_ablation = _model_ablation(validation_predictions)
    validation_anchor = pd.read_csv(
        cfg.path("results_dir") / "validation_predictions" / f"{anchor_model}.csv",
        index_col="date",
    )["forecast"]
    validation_reconciled_predictions = {
        name: reconcile_to_daily_means(prediction, validation_anchor, horizon=24)
        for name, prediction in validation_predictions.items()
    }
    validation_reconciled_ablation = _model_ablation(validation_reconciled_predictions)
    selected_model = frozen_selection[0] if frozen_selection else str(validation_reconciled_ablation.index[0])

    evaluation_models = _fit_models(cfg, long_frame, train_end=cfg.split.val_end)
    model = evaluation_models[selected_model]
    daily_anchor_estimator, daily_matrix, daily_columns = _fit_frozen_daily_anchor(
        cfg,
        anchor_model,
    )
    calibration = _predict_period(
        model,
        long_frame,
        start=cfg.split.calibration_start,
        end=cfg.split.calibration_end,
    )
    test = _predict_period(
        model,
        long_frame,
        start=cfg.split.test_start,
        end=cfg.split.test_end,
    )
    ablation_predictions = {
        name: (
            test
            if name == selected_model
            else _predict_period(
                candidate,
                long_frame,
                start=cfg.split.test_start,
                end=cfg.split.test_end,
            )
        )
        for name, candidate in evaluation_models.items()
    }
    calibration_anchor = _predict_daily_anchors(
        daily_anchor_estimator,
        daily_matrix,
        daily_columns,
        start=cfg.split.calibration_start,
        end=cfg.split.calibration_end,
    )
    test_anchor = _predict_daily_anchors(
        daily_anchor_estimator,
        daily_matrix,
        daily_columns,
        start=cfg.split.test_start,
        end=cfg.split.test_end,
    )
    test_reconciled_predictions = {
        name: reconcile_to_daily_means(prediction, test_anchor, horizon=24)
        for name, prediction in ablation_predictions.items()
    }
    reconciled_calibration = reconcile_to_daily_means(
        calibration,
        calibration_anchor,
        horizon=24,
    )
    reconciled_test = test_reconciled_predictions[selected_model]
    intervals = _restore_interval_identity(
        horizon_conformal_intervals(
        reconciled_calibration.reset_index(drop=True),
        reconciled_test.reset_index(drop=True),
        alpha=0.1,
        ),
        reconciled_test,
    )

    quantile_train_mask = _date_mask(
        long_frame["valid_time"],
        cfg.dataset_start,
        cfg.split.val_end,
    ) & (long_frame["horizon"] == 24)
    quantile_frame = long_frame[long_frame["horizon"] == 24]
    quantile_models = {
        name: make_hourly_quantile_lightgbm(cfg, quantile=quantile).fit(
            long_frame[quantile_train_mask]
        )
        for name, quantile in (("lower", 0.05), ("upper", 0.95))
    }
    config_path = Path(args.config).resolve()
    protocol_records = _hourly_protocol_records(
        cfg,
        selected_model=selected_model,
        point_model=model,
        daily_anchor_model=anchor_model,
        daily_columns=daily_columns,
        quantile_models=quantile_models,
        config_path=config_path,
    )
    selected_point_record = protocol_records[f"hourly/point/{selected_model}"]
    for name in ablation_predictions:
        stream_id = f"hourly/point/{name}"
        protocol_records.setdefault(
            stream_id,
            {
                **selected_point_record,
                "stream_id": stream_id,
                "model_identity": f"direct_hourly:{name}",
                "model_parameters": {"selected_model": name, "horizon": 24},
            },
        )
    merge_protocol_manifest(
        artifact_results_root(cfg) / "evaluation_protocol.json",
        protocol_records,
        "hourly/",
    )
    point_record = protocol_records[f"hourly/point/{selected_model}"]
    quantile_record = protocol_records["hourly/quantile/lightgbm_cqr"]
    point_calibration_metadata = _protocol_metadata(point_record, "calibration")
    point_final_metadata = _protocol_metadata(point_record, "retrospective_final")
    quantile_calibration_metadata = _protocol_metadata(quantile_record, "calibration")
    quantile_final_metadata = _protocol_metadata(quantile_record, "retrospective_final")
    assert_compatible_artifacts(point_calibration_metadata, point_final_metadata)
    assert_compatible_artifacts(quantile_calibration_metadata, quantile_final_metadata)
    quantile_calibration = {
        name: _predict_period(
            candidate,
            quantile_frame,
            start=cfg.split.calibration_start,
            end=cfg.split.calibration_end,
        )
        for name, candidate in quantile_models.items()
    }
    quantile_test = {
        name: _predict_period(
            candidate,
            quantile_frame,
            start=cfg.split.test_start,
            end=cfg.split.test_end,
        )
        for name, candidate in quantile_models.items()
    }
    reconciled_calibration_h24 = reconciled_calibration[reconciled_calibration["horizon"] == 24]
    reconciled_test_h24 = reconciled_test[reconciled_test["horizon"] == 24]
    cqr_calibration = _restore_interval_identity(
        _reconciled_quantile_frame(
            reconciled_calibration_h24,
            quantile_calibration["lower"],
            quantile_calibration["upper"],
        ),
        reconciled_calibration_h24,
    )
    cqr_input = _reconciled_quantile_frame(
        reconciled_test_h24,
        quantile_test["lower"],
        quantile_test["upper"],
    )
    cqr_intervals = _restore_interval_identity(
        horizon_cqr_intervals(cqr_calibration, cqr_input, alpha=0.1),
        cqr_input,
    )

    drift_features = [
        "Temp_forecast",
        "Wind_forecast",
        "HDD",
        "CDD",
        "L_t-24",
        "L_t-168",
    ]
    reference = features[
        _date_mask(features.index.to_series(), cfg.dataset_start, cfg.split.val_end)
    ]
    current = features[
        _date_mask(features.index.to_series(), cfg.split.test_start, cfg.split.test_end)
    ]
    current_months = set(current.index.tz_convert("Europe/Berlin").month)
    reference = reference[reference.index.tz_convert("Europe/Berlin").month.isin(current_months)]
    drift = feature_drift_report(reference, current, drift_features)
    calibration_scale = max(float(calibration["error"].abs().std()), 1.0)
    residual_drift = page_hinkley(
        test["error"].abs().to_numpy(),
        delta=0.01 * calibration_scale,
        threshold=5.0 * calibration_scale,
        reference_mean=float(calibration["error"].abs().mean()),
    )
    adaptive_lower, adaptive_upper, _ = adaptive_conformal_interval(
        reconciled_calibration["actual"].to_numpy(dtype="float64"),
        reconciled_calibration["prediction"].to_numpy(dtype="float64"),
        reconciled_test["actual"].to_numpy(dtype="float64"),
        reconciled_test["prediction"].to_numpy(dtype="float64"),
        alpha=0.1,
        gamma=float(cfg.uncertainty["adaptive_gamma"]),
        window=int(cfg.uncertainty["adaptive_window"]),
    )
    adaptive_intervals = reconciled_test.assign(
        lower=adaptive_lower,
        upper=adaptive_upper,
    )
    point_evidence = _interval_evidence_rows(
        {"symmetric": intervals, "adaptive": adaptive_intervals},
        metadata=point_final_metadata,
        alpha=0.1,
    )
    cqr_evidence = _interval_evidence_rows(
        {"cqr": cqr_intervals},
        metadata=quantile_final_metadata,
        alpha=0.1,
    )
    interval_evidence_output = pd.concat([point_evidence, cqr_evidence], ignore_index=True)
    monthly_interval_output = pd.concat(
        [
            _monthly_interval_evidence_rows(
                {"symmetric": intervals, "adaptive": adaptive_intervals},
                metadata=point_final_metadata,
                alpha=0.1,
            ),
            _monthly_interval_evidence_rows(
                {"cqr": cqr_intervals},
                metadata=quantile_final_metadata,
                alpha=0.1,
            ),
        ],
        ignore_index=True,
    )
    monthly_residual_output = _monthly_residual_drift(
        test,
        residual_drift,
        metadata=point_final_metadata,
    )

    output = artifact_results_root(cfg) / "hourly"
    output.mkdir(parents=True, exist_ok=True)
    for name, prediction in validation_predictions.items():
        _write_hourly_artifact(
            output / "validation_predictions" / f"{name}.csv",
            prediction,
            dataset,
            weather_strategy=weather_strategy,
        )
    for name, prediction in validation_reconciled_predictions.items():
        _write_hourly_artifact(
            output / "validation_reconciled_predictions" / f"{name}.csv",
            prediction,
            dataset,
            weather_strategy=weather_strategy,
        )
    for name, prediction in ablation_predictions.items():
        record = protocol_records.get(f"hourly/point/{name}")
        if record is None:
            raise ValueError(f"missing protocol record for hourly/point/{name}")
        _write_hourly_artifact(
            output / "test_predictions" / f"{name}.csv",
            prediction,
            dataset,
            weather_strategy=weather_strategy,
            metadata=_protocol_metadata(record, "retrospective_final"),
        )
    for name, prediction in test_reconciled_predictions.items():
        _write_hourly_artifact(
            output / "test_reconciled_predictions" / f"{name}.csv",
            prediction,
            dataset,
            weather_strategy=weather_strategy,
        )
    _write_hourly_artifact(
        output / "calibration_predictions.csv",
        calibration,
        dataset,
        weather_strategy=weather_strategy,
        metadata=point_calibration_metadata,
    )
    _write_hourly_artifact(
        output / "test_intervals.csv",
        intervals,
        dataset,
        weather_strategy=weather_strategy,
        metadata=point_final_metadata,
    )
    _write_hourly_artifact(
        output / "cqr_calibration_predictions.csv",
        cqr_calibration,
        dataset,
        weather_strategy=weather_strategy,
        metadata=quantile_calibration_metadata,
    )
    _write_hourly_artifact(
        output / "cqr_test_intervals.csv",
        cqr_intervals,
        dataset,
        weather_strategy=weather_strategy,
        metadata=quantile_final_metadata,
    )
    model_metrics = metrics_by_horizon(test).assign(model=selected_model)
    naive_metrics = metrics_by_horizon(test.assign(error=test["seasonal_naive_error"])).assign(
        model="seasonal_naive_168h"
    )
    pd.concat([model_metrics, naive_metrics]).reset_index().set_index(["model", "horizon"]).to_csv(
        output / "metrics_by_horizon.csv"
    )
    interval_evidence_output.query("slice_type == 'aggregate'").to_csv(
        output / "interval_comparison.csv", index=False
    )
    interval_evidence_output.query("slice_type == 'horizon'").to_csv(
        output / "coverage_by_horizon.csv", index=False
    )
    interval_evidence_output.query("slice_type == 'local_day'").to_csv(
        output / "coverage_by_local_day.csv", index=False
    )
    monthly_interval_output.to_csv(output / "coverage_by_month.csv", index=False)
    _local_hour_evidence_rows(
        cqr_intervals,
        metadata=quantile_final_metadata,
        alpha=0.1,
    ).to_csv(output / "cqr_coverage_by_hour.csv", index=False)
    drift.to_csv(output / "feature_drift.csv")
    residual_drift.to_csv(output / "residual_drift.csv", index=False)
    monthly_residual_output.to_csv(output / "residual_drift_monthly.csv", index=False)
    _write_hourly_artifact(
        output / "test_reconciled_predictions.csv",
        reconciled_test,
        dataset,
        weather_strategy=weather_strategy,
        metadata=point_final_metadata,
    )
    validation_ablation.to_csv(output / "model_ablation_validation.csv")
    _model_ablation(ablation_predictions).to_csv(output / "model_ablation_test.csv")
    validation_reconciled_ablation.to_csv(output / "model_ablation_validation_reconciled.csv")
    _model_ablation(test_reconciled_predictions).to_csv(
        output / "model_ablation_test_reconciled.csv"
    )
    _temporal_model_ablation(validation_predictions).to_csv(
        output / "model_ablation_validation_monthly.csv"
    )
    _temporal_model_ablation(ablation_predictions).to_csv(
        output / "model_ablation_test_monthly.csv"
    )
    _temporal_model_ablation(validation_reconciled_predictions).to_csv(
        output / "model_ablation_validation_reconciled_monthly.csv"
    )
    _temporal_model_ablation(test_reconciled_predictions).to_csv(
        output / "model_ablation_test_reconciled_monthly.csv"
    )
    _paired_daily_mae_bootstrap(
        test_reconciled_predictions,
        reference=selected_model,
        seed=cfg.seed,
    ).to_csv(output / "model_ablation_test_uncertainty.csv")
    if frozen_selection is None:
        pd.DataFrame(
            [
                {
                    "selected_model": selected_model,
                    "daily_anchor_model": anchor_model,
                    "selection_period_start": cfg.split.train_end + timedelta(days=1),
                    "selection_period_end": cfg.split.val_end,
                    "selection_metric": "reconciled_h24_hourly_MAE",
                    "selection_evidence_period": "validation",
                    "selection_protocol_fingerprint": point_final_metadata["protocol_fingerprint"],
                }
            ]
        ).to_csv(output / "model_selection.csv", index=False)
    _reconciliation_comparison(
        test,
        reconciled_test,
        selected_model=selected_model,
        anchor_model=anchor_model,
        metadata={
            key: point_final_metadata[key]
            for key in ("evaluation_period", "stream_id", "protocol_fingerprint")
        },
    ).to_csv(output / "reconciliation_metrics.csv")
    reconciliation_invariant_rows(reconciled_test).to_csv(
        output / "reconciliation_invariants.csv", index=False
    )
    _daily_model_comparison(
        cfg,
        test,
        reconciled_test,
        selected_model=selected_model,
        anchor_model=anchor_model,
    ).to_csv(output / "daily_model_comparison.csv")
    print(f"wrote hourly evaluation to {output}")


if __name__ == "__main__":
    main()
