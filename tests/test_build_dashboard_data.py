from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from loadfc.evaluation.protocol import protocol_fingerprint
from loadfc.tracking import sha256_file
from scripts import build_dashboard_data
from scripts.build_dashboard_data import main
from scripts.validate_results import _validate_release_manifest
from scripts.write_run_summary import _release_identity

ROOT = Path(__file__).resolve().parents[1]


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _record(source_revision: str, config_sha256: str, stream_id: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "source_revision": source_revision,
        "config_sha256": config_sha256,
        "stream_id": stream_id,
        "model_identity": stream_id.rsplit("/", 1)[-1],
        "ordered_feature_columns": ["lag_1"],
        "model_parameters": {"seed": 42},
        "seed": 42,
        "splits": {
            "train": ["2019-01-14", "2023-12-31"],
            "validation": ["2024-01-01", "2025-06-30"],
            "calibration": ["2025-07-01", "2025-12-31"],
            "retrospective_final": ["2026-01-01", "2026-08-04"],
        },
        "weather_strategy": "available_day_ahead",
        "point_state_policy": {"fit_through": "2025-06-30", "update": "none"},
        "interval_state_policy": {"fixed": "calibration_scores_frozen"},
        "final_role": "retrospective_final",
        "validation_selection_evidence": "results/metrics/validation_metrics.csv",
        "rationale": "validation-selected fixture",
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _release_tree(tmp_path: Path) -> tuple[Path, Path, dict[str, dict[str, object]]]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Release Test")
    _git(repo, "config", "user.email", "release@example.invalid")
    (repo / "config.yaml").write_bytes((ROOT / "config.yaml").read_bytes())
    (repo / "uv.lock").write_bytes((ROOT / "uv.lock").read_bytes())
    (repo / ".gitignore").write_text("/results/\n/.staging/\n")
    _git(repo, "add", ".gitignore", "config.yaml", "uv.lock")
    _git(repo, "commit", "-qm", "fixture")

    staging = repo / ".staging"
    staging.mkdir()
    config_path = staging / "config.yaml"
    config_path.write_bytes((repo / "config.yaml").read_bytes())
    (staging / "report").mkdir()
    (staging / "report/generated_results.tex").write_text("generated\n")
    results = staging / "results"
    results.mkdir()

    revision = _git(repo, "rev-parse", "HEAD")
    config_hash = sha256_file(config_path)
    daily_models = [
        "SARIMAX",
        "ensemble",
        "lightgbm",
        "naive_1d",
        "random_forest",
        "seasonal_naive_7d",
        "xgboost",
    ]
    daily_records = {
        model: _record(revision, config_hash, f"daily/{model}") for model in daily_models
    }
    daily = daily_records["ensemble"]
    hourly = _record(revision, config_hash, "hourly/point/residual_hybrid")
    hourly_candidate = _record(revision, config_hash, "hourly/point/lightgbm_direct")
    hourly_ridge = _record(revision, config_hash, "hourly/point/ridge_direct")
    quantile = _record(revision, config_hash, "hourly/quantile/lightgbm_cqr")
    records = {
        record["stream_id"]: record
        for record in (
            *daily_records.values(),
            hourly,
            hourly_candidate,
            hourly_ridge,
            quantile,
        )
    }
    fingerprints = {name: protocol_fingerprint(record) for name, record in records.items()}
    (results / "evaluation_protocol.json").write_text(
        json.dumps({"schema_version": 1, "records": records})
    )

    daily_fingerprint = fingerprints["daily/ensemble"]
    _write_csv(
        results / "metrics/daily_comparison.csv",
        [
            {
                "model": model,
                "evaluation_period": "retrospective_final",
                "stream_id": f"daily/{model}",
                "protocol_fingerprint": fingerprints[f"daily/{model}"],
                "dates": 7,
                "start": "2026-01-01",
                "end": "2026-01-07",
                "n": 7,
                "RMSE": rmse,
                "MAE": mae,
                "MAPE": mape,
                "MASE": mase,
            }
            for model, rmse, mae, mape, mase in (
                ("SARIMAX", 125.0, 105.0, 2.1, 0.75),
                ("ensemble", 120.0, 100.0, 2.0, 0.7),
                ("lightgbm", 130.0, 110.0, 2.2, 0.8),
                ("naive_1d", 180.0, 150.0, 3.0, 1.0),
                ("random_forest", 135.0, 115.0, 2.3, 0.82),
                ("seasonal_naive_7d", 200.0, 170.0, 3.4, 1.1),
                ("xgboost", 128.0, 108.0, 2.15, 0.78),
            )
        ],
    )
    daily_metadata = {
        "stream_id": "daily/ensemble",
        "protocol_fingerprint": daily_fingerprint,
        "evaluation_period": "retrospective_final",
        "point_state_policy": json.dumps(daily["point_state_policy"], sort_keys=True),
        "interval_state_policy": json.dumps(daily["interval_state_policy"], sort_keys=True),
    }
    prediction_rows = [
        {
            "date": f"2026-01-{day:02d}",
            "actual": 1000.0 + day,
            "forecast": 990.0 + day,
            "forecast_origin": f"2025-12-{day + 30:02d}" if day == 1 else f"2026-01-{day - 1:02d}",
            "valid_time": f"2026-01-{day:02d}",
            "weather_source_run": "open_meteo_previous_day1",
            "weather_availability_assumption": "previous_day1 available 24 hours before valid_time",
            **daily_metadata,
        }
        for day in range(1, 8)
    ]
    _write_csv(results / "predictions/ensemble.csv", prediction_rows)
    _write_csv(
        results / "interval_predictions/ensemble.csv",
        [
            {
                **row,
                "fixed_lower_80": row["forecast"] - 50.0,
                "fixed_upper_80": row["forecast"] + 50.0,
                "lower_80": row["forecast"] - 55.0,
                "upper_80": row["forecast"] + 55.0,
                "fixed_lower_95": row["forecast"] - 80.0,
                "fixed_upper_95": row["forecast"] + 80.0,
                "lower_95": row["forecast"] - 90.0,
                "upper_95": row["forecast"] + 90.0,
            }
            for row in prediction_rows
        ],
    )
    _write_csv(
        results / "metrics/interval_coverage.csv",
        [
            {
                "model": "ensemble",
                "method": method,
                "level": level,
                **daily_metadata,
                "coverage_scope": (
                    "empirical_retrospective"
                    if method == "fixed"
                    else "prequential_monitoring_no_unconditional_time_series_coverage"
                ),
                "nominal": nominal,
                "empirical_coverage": coverage,
                "mean_width_MW": width,
                "interval_score_MW": score,
                "n": 7,
            }
            for method, level, nominal, coverage, width, score in (
                ("fixed", "80%", 0.8, 0.857142857, 100.0, 120.0),
                ("fixed", "95%", 0.95, 1.0, 160.0, 160.0),
                ("adaptive", "80%", 0.8, 0.857142857, 110.0, 130.0),
                ("adaptive", "95%", 0.95, 1.0, 180.0, 180.0),
            )
        ],
    )

    hourly_fingerprint = fingerprints["hourly/point/residual_hybrid"]
    _write_csv(
        results / "hourly/model_selection.csv",
        [
            {
                "selected_model": "residual_hybrid",
                "daily_anchor_model": "ensemble",
                "selection_period_start": "2024-01-01",
                "selection_period_end": "2025-06-30",
                "selection_metric": "reconciled_h24_hourly_MAE",
                "selection_evidence_period": "validation",
                "selection_protocol_fingerprint": hourly_fingerprint,
            }
        ],
    )
    for name, base_mae in (
        ("model_ablation_validation.csv", 200.0),
        ("model_ablation_validation_reconciled.csv", 180.0),
        ("model_ablation_test.csv", 210.0),
        ("model_ablation_test_reconciled.csv", 190.0),
    ):
        _write_csv(
            results / f"hourly/{name}",
            [
                {
                    "model": model,
                    "hourly_MAE": base_mae + model_index,
                    "hourly_RMSE": 250.0,
                    "hourly_MAPE": 2.5,
                    "daily_MAE": 100.0,
                    "daily_RMSE": 120.0,
                    "daily_MAPE": 2.0,
                    "n_hours": 168,
                    "n_days": 7,
                }
                for model_index, model in enumerate(
                    ("residual_hybrid", "lightgbm_direct", "ridge_direct")
                )
            ],
        )
    _write_csv(
        results / "hourly/interval_comparison.csv",
        [
            {
                "method": method,
                "level": "90%",
                "slice_type": "aggregate",
                "slice_value": "all",
                "evaluation_period": "retrospective_final",
                "coverage_scope": scope,
                "stream_id": stream_id,
                "protocol_fingerprint": fingerprint,
                "nominal": 0.9,
                "empirical_coverage": coverage,
                "mean_width": width,
                "interval_score": score,
                "n": 191,
            }
            for method, scope, stream_id, fingerprint, coverage, width, score in (
                (
                    "symmetric",
                    "empirical_retrospective",
                    "hourly/point/residual_hybrid",
                    hourly_fingerprint,
                    0.88,
                    300.0,
                    350.0,
                ),
                (
                    "adaptive",
                    "prequential_monitoring_no_unconditional_time_series_coverage",
                    "hourly/point/residual_hybrid",
                    hourly_fingerprint,
                    0.9,
                    320.0,
                    340.0,
                ),
                (
                    "cqr",
                    "empirical_retrospective",
                    "hourly/quantile/lightgbm_cqr",
                    fingerprints["hourly/quantile/lightgbm_cqr"],
                    0.89,
                    330.0,
                    360.0,
                ),
            )
        ],
    )

    hourly_rows: list[dict[str, object]] = []
    for model, stream_id in (
        ("residual_hybrid", "hourly/point/residual_hybrid"),
        ("lightgbm_direct", "hourly/point/lightgbm_direct"),
        ("ridge_direct", "hourly/point/ridge_direct"),
    ):
        fingerprint = fingerprints[stream_id]
        for slice_type, slice_value, n in (
            ("aggregate", "all", 191),
            ("horizon", 24, 191),
            ("local_hour", 0, 8),
            ("local_hour", 1, 8),
        ):
            hourly_rows.append(
                {
                    "model": model,
                    "slice_type": slice_type,
                    "slice_value": slice_value,
                    "evaluation_period": "retrospective_final",
                    "stream_id": stream_id,
                    "protocol_fingerprint": fingerprint,
                    "eligible_horizon": "24",
                    "local_start": "2026-01-01T00:00:00+01:00",
                    "local_end": "2026-01-08T22:00:00+01:00",
                    "n": n,
                    "RMSE": 250.0,
                    "MAE": 190.0 if model == "residual_hybrid" else 195.0,
                    "MAPE": 2.5,
                    "MASE": 0.8,
                }
            )
    _write_csv(results / "hourly/hourly_comparison.csv", hourly_rows)
    _write_csv(
        results / "hourly/metrics_by_horizon.csv",
        [
            {
                "model": model,
                "horizon": horizon,
                "mae": mae + horizon,
                "rmse": 250.0 + horizon,
                "n": 191,
            }
            for model, mae in (
                ("residual_hybrid", 190.0),
                ("lightgbm_direct", 195.0),
                ("ridge_direct", 230.0),
            )
            for horizon in range(1, 25)
        ],
    )

    utc_times = pd.date_range("2025-12-31T23:00:00Z", periods=192, freq="h")[:-1]
    for model, stream_id, offset in (
        ("residual_hybrid", "hourly/point/residual_hybrid", 0.0),
        ("lightgbm_direct", "hourly/point/lightgbm_direct", 5.0),
        ("ridge_direct", "hourly/point/ridge_direct", 10.0),
    ):
        fingerprint = fingerprints[stream_id]
        raw_rows = [
            {
                "prediction": 1000.0 + position + offset,
                "actual": 1010.0 + position,
                "forecast_origin": (timestamp - pd.Timedelta(hours=24)).isoformat(),
                "valid_time": timestamp.isoformat(),
                "horizon": 24,
                "stream_id": stream_id,
                "protocol_fingerprint": fingerprint,
                "evaluation_period": "retrospective_final",
                "point_state_policy": json.dumps(hourly["point_state_policy"], sort_keys=True),
                "interval_state_policy": json.dumps(
                    hourly["interval_state_policy"], sort_keys=True
                ),
                "weather_availability_assumption": "previous_day1 available before valid_time",
            }
            for position, timestamp in enumerate(utc_times)
        ]
        reconciled_rows = [
            {
                "prediction": row["prediction"] - 2.0,
                "actual": row["actual"],
                "forecast_origin": row["forecast_origin"],
                "valid_time": row["valid_time"],
                "horizon": 24,
            }
            for row in raw_rows
        ]
        _write_csv(results / f"hourly/test_predictions/{model}.csv", raw_rows)
        _write_csv(results / f"hourly/test_reconciled_predictions/{model}.csv", reconciled_rows)

    _write_csv(
        results / "hourly/reconciliation_metrics.csv",
        [
            {
                "model": model,
                "MAE": mae,
                "RMSE": 250.0,
                "MAPE": 2.5,
                "n_hours": 191,
                "evaluation_period": "retrospective_final",
                "stream_id": "hourly/point/residual_hybrid",
                "protocol_fingerprint": hourly_fingerprint,
            }
            for model, mae in (
                ("hourly_residual_hybrid_h24", 210.0),
                ("hourly_residual_hybrid_h24_ensemble_reconciled", 190.0),
            )
        ],
    )
    _write_csv(
        results / "hourly/reconciliation_invariants.csv",
        [
            {
                "local_date": f"2026-01-{day:02d}",
                "reconciled_hourly_mean": 1000.0,
                "daily_anchor": 1000.0,
                "delta": 0.0,
                "abs_delta": 0.0,
                "tolerance": 0.000001,
                "pass": True,
                "n": 24 if day < 8 else 23,
            }
            for day in range(1, 9)
        ],
    )
    _write_csv(
        results / "hourly/model_ablation_test_uncertainty.csv",
        [
            {
                "candidate": "lightgbm_direct",
                "reference": "residual_hybrid",
                "mae_difference": 5.0,
                "ci_lower": -10.0,
                "ci_upper": 20.0,
                "probability_candidate_better": 0.4,
                "n_days": 7,
                "seed": 42,
                "n_bootstrap": 1000,
                "block_size_days": 7,
                "practical_tie_threshold": 50.0,
                "practical_tie": True,
                "selected_model": "residual_hybrid",
                "selected_model_preserved": True,
                "uncertainty_scope": "retrospective_final time-series block bootstrap",
                "validation_selection_metric": "reconciled_h24_hourly_MAE",
                "validation_selected_model": "residual_hybrid",
                "validation_selection_evidence": (
                    "results/hourly/model_ablation_validation_reconciled.csv"
                ),
            }
        ],
    )

    descriptive_scope = "full_available_history_descriptive"
    _write_csv(
        results / "analysis/load_profile_weekday.csv",
        [
            {
                "weekday_order": order,
                "weekday": weekday,
                "mean_load_MW": 50_000.0 + order,
                "n_days": 52,
                "evidence_scope": descriptive_scope,
            }
            for order, weekday in enumerate(
                [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday",
                ],
                start=1,
            )
        ],
    )
    _write_csv(
        results / "analysis/load_profile_month.csv",
        [
            {
                "month_order": order,
                "month": month,
                "mean_load_MW": 50_000.0 + order,
                "n_days": 30,
                "evidence_scope": descriptive_scope,
            }
            for order, month in enumerate(
                [
                    "January",
                    "February",
                    "March",
                    "April",
                    "May",
                    "June",
                    "July",
                    "August",
                    "September",
                    "October",
                    "November",
                    "December",
                ],
                start=1,
            )
        ],
    )
    _write_csv(
        results / "analysis/temperature_load_curve.csv",
        [
            {
                "bin_order": order,
                "lower_C": lower,
                "upper_C": lower + 5,
                "mean_load_MW": 50_000.0 + order,
                "n_days": 10,
                "evidence_scope": descriptive_scope,
            }
            for order, lower in enumerate(range(-10, 35, 5), start=1)
        ],
    )
    _write_csv(
        results / "analysis/shap_importance.csv",
        [
            {
                "feature": feature,
                "xgboost": value,
                "lightgbm": value - 1.0,
                "mean": value - 0.5,
            }
            for feature, value in (("L_t-7", 30.0), ("L_t-1", 20.0), ("Weekend", 10.0))
        ],
    )

    rolling = pd.DataFrame(
        {
            model: [2.0 + period_index / 10 + model_index / 100 for period_index in range(4)]
            for model_index, model in enumerate(
                [
                    "SARIMAX",
                    "xgboost",
                    "lightgbm",
                    "random_forest",
                    "naive_1d",
                    "seasonal_naive_7d",
                ]
            )
        },
        index=["2024-H1", "2024-H2", "2025-H1", "2025-H2"],
    )
    rolling.index.name = "period"
    rolling_path = results / "metrics/rolling_origin_mape.csv"
    rolling_path.parent.mkdir(parents=True, exist_ok=True)
    rolling.to_csv(rolling_path)
    _write_csv(
        results / "metrics/generalization_gap.csv",
        [
            {
                "model": model,
                "validation_MAPE": 2.0 + index / 10,
                "test_MAPE": 2.1 + index / 10,
                "absolute_gap_pp": 0.1,
                "relative_gap_pct": 5.0,
            }
            for index, model in enumerate(daily_models)
        ],
    )
    error_rows = []
    for role, months in (
        ("validation", [f"2024-{month:02d}" for month in range(1, 13)] + [f"2025-{month:02d}" for month in range(1, 7)]),
        ("test", [f"2026-{month:02d}" for month in range(1, 9)]),
    ):
        for model in daily_models:
            for month in months:
                error_rows.append(
                    {
                        "period": role,
                        "model": model,
                        "slice_type": "month",
                        "slice": month,
                        "n": 30,
                        "MAE": 100.0,
                        "MAPE": 2.0,
                        "bias_MW": 1.0,
                    }
                )
            for day_type in ("weekday", "weekend"):
                error_rows.append(
                    {
                        "period": role,
                        "model": model,
                        "slice_type": "day_type",
                        "slice": day_type,
                        "n": 100,
                        "MAE": 100.0,
                        "MAPE": 2.0,
                        "bias_MW": 1.0,
                    }
                )
    _write_csv(results / "metrics/error_slices.csv", error_rows)
    _write_csv(
        results / "hourly/feature_drift.csv",
        [
            {"feature": feature, "psi": psi, "status": status}
            for feature, psi, status in (
                ("Temp_forecast", 0.05, "stable"),
                ("Wind_forecast", 0.3, "critical"),
                ("HDD", 0.04, "stable"),
                ("CDD", 0.03, "stable"),
                ("L_t-24", 0.02, "stable"),
                ("L_t-168", 0.01, "stable"),
            )
        ],
    )

    interval_months = [f"2026-{month:02d}" for month in range(1, 9)]
    monthly_coverage = []
    horizon_coverage = []
    for method, stream_id in (
        ("symmetric", "hourly/point/residual_hybrid"),
        ("adaptive", "hourly/point/residual_hybrid"),
        ("cqr", "hourly/quantile/lightgbm_cqr"),
    ):
        coverage_scope = (
            "prequential_monitoring_no_unconditional_time_series_coverage"
            if method == "adaptive"
            else "empirical_retrospective"
        )
        base = {
            "method": method,
            "level": "90%",
            "evaluation_period": "retrospective_final",
            "coverage_scope": coverage_scope,
            "stream_id": stream_id,
            "protocol_fingerprint": fingerprints[stream_id],
            "nominal": 0.9,
            "empirical_coverage": 0.9,
            "mean_width": 300.0,
            "interval_score": 350.0,
            "n": 191,
        }
        monthly_coverage.extend(
            {**base, "slice_type": "month", "slice_value": month}
            for month in interval_months
        )
        horizon_coverage.append({**base, "slice_type": "horizon", "slice_value": 24})
    _write_csv(results / "hourly/coverage_by_month.csv", monthly_coverage)
    _write_csv(results / "hourly/coverage_by_horizon.csv", horizon_coverage)
    _write_csv(
        results / "hourly/residual_drift_monthly.csv",
        [
            {
                "month": month,
                "evaluation_period": "retrospective_final",
                "monitoring_scope": "prequential_monitoring",
                "stream_id": "hourly/point/residual_hybrid",
                "protocol_fingerprint": hourly_fingerprint,
                "mean_absolute_error_MW": 190.0,
                "max_page_hinkley_statistic": 12.0,
                "alert_count": 1,
                "n": 191,
            }
            for month in interval_months
        ],
    )
    _write_csv(
        results / "hourly/residual_drift.csv",
        [
            {"value": 1.0, "running_mean": 1.0, "statistic": 1.0, "alert": False}
            for _ in range(191 * len(interval_months))
        ],
    )

    (results / "metrics/low_risk_improvement_decision.json").write_text(
        json.dumps(
            {
                "candidate": "persistence_weather",
                "criterion": "ensemble_MAPE",
                "decision": "accepted",
                "rationale": "validation-only improvement",
                "selected_model": "ensemble",
                "validation_period_start": "2024-01-01",
                "validation_period_end": "2025-06-30",
                "validation_evidence_path": (
                    "results/metrics/persistence_weather_validation_metrics.csv"
                ),
                "integrity_checks": {"no_final_outcomes_used": True},
            }
        )
    )

    identity = _release_identity(config_path, results, fingerprints, artifact_root=staging)
    summary = {
        "schema_version": 2,
        "generated_at_utc": "2026-08-09T00:00:00+00:00",
        "final_role": "retrospective_final",
        "retrospective_final_metrics": {
            "ensemble": {"MAE": 100.0, "RMSE": 120.0, "MAPE": 2.0, "MASE": 0.7}
        },
        **identity,
    }
    (results / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return repo, staging, records


def _run(config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["build_dashboard_data.py", "--config", str(config_path)])
    main()


def _refresh_identity(staging: Path, records: dict[str, dict[str, object]]) -> None:
    summary_path = staging / "results/run_summary.json"
    summary = json.loads(summary_path.read_text())
    fingerprints = {name: protocol_fingerprint(record) for name, record in records.items()}
    summary.update(
        _release_identity(
            staging / "config.yaml",
            staging / "results",
            fingerprints,
            artifact_root=staging,
        )
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def test_daily_tracer_generates_deterministic_declared_release(tmp_path, monkeypatch):
    repo, staging, records = _release_tree(tmp_path)
    destination = staging / "dashboard/src/generated/release.json"

    _run(staging / "config.yaml", monkeypatch)
    first = destination.read_bytes()
    payload = json.loads(first)
    _run(staging / "config.yaml", monkeypatch)

    assert destination.read_bytes() == first
    assert payload["schema_version"] == 1
    assert payload["source_revision"] == _git(repo, "rev-parse", "HEAD")
    assert payload["generated_at"] == "2026-08-09T00:00:00+00:00"
    assert payload["final_role"] == "retrospective_final"
    assert payload["selector_options"]["frequency"] == ["daily", "hourly"]
    assert payload["selector_options"]["daily_models"] == ["ensemble"]
    assert len(payload["overview"]) == 3
    assert payload["forecast"]["daily"]["model"] == "ensemble"
    assert len(payload["forecast"]["daily"]["rows"]) == 7
    assert {row["method"] for row in payload["intervals"]["daily"]} == {
        "adaptive",
        "fixed",
    }

    summary = json.loads((staging / "results/run_summary.json").read_text())
    assert summary["declared_artifacts"]["dashboard/src/generated/release.json"] == sha256_file(
        destination
    )
    assert payload["bundle_fingerprint"] == summary["bundle_fingerprint"]
    _validate_release_manifest(
        summary,
        staging / "results",
        records,
        repo,
        staging / "config.yaml",
    )


def test_complete_dashboard_contract_is_bounded_and_uses_last_complete_week(
    tmp_path, monkeypatch
):
    _, staging, _ = _release_tree(tmp_path)

    _run(staging / "config.yaml", monkeypatch)

    destination = staging / "dashboard/src/generated/release.json"
    payload = json.loads(destination.read_text())
    assert destination.stat().st_size <= 2 * 1024 * 1024
    assert set(payload) >= {
        "forecast",
        "intervals",
        "hourly_error",
        "reconciliation",
        "comparison",
        "protocol",
        "limitations",
    }
    assert payload["selector_options"]["hourly_models"] == [
        "lightgbm_direct",
        "residual_hybrid",
        "ridge_direct",
    ]
    assert payload["selector_options"]["hourly_point_error_horizons"] == list(range(1, 25))
    assert payload["selector_options"]["hourly_interval_coverage_horizons"] == [24]
    assert payload["selector_options"]["hourly_point_error_models"] == [
        "lightgbm_direct",
        "residual_hybrid",
        "ridge_direct",
    ]
    assert payload["selector_options"]["local_hours"] == [0, 1]
    assert payload["selector_options"]["output_states"] == ["raw", "reconciled"]
    assert payload["selector_options"]["hourly_interval_methods"] == [
        "adaptive",
        "cqr",
        "symmetric",
    ]
    hourly = payload["forecast"]["hourly"]
    assert hourly["window"] == {
        "local_start": "2026-01-01",
        "local_end": "2026-01-07",
        "local_days": 7,
    }
    assert len(hourly["rows"]) == 3 * 2 * 7 * 24
    assert {row["state"] for row in hourly["rows"]} == {"raw", "reconciled"}
    assert {row["model"] for row in hourly["rows"]} == {
        "lightgbm_direct",
        "residual_hybrid",
        "ridge_direct",
    }
    assert all(row["horizon"] == 24 for row in hourly["rows"])
    assert {row["slice_type"] for row in payload["hourly_error"]["rows"]} == {
        "aggregate",
        "horizon",
        "local_hour",
    }
    assert payload["hourly_error"]["dst_sensitive_sample_sizes"] is True
    assert payload["reconciliation"]["invariant"]["all_passed"] is True
    assert payload["reconciliation"]["invariant"]["complete_local_days"] == 7
    assert payload["comparison"]["selection"]["selection_evidence_period"] == "validation"
    assert payload["comparison"]["bootstrap"][0]["practical_tie"] is True
    assert payload["comparison"]["low_risk_decision"]["decision"] == "accepted"
    assert payload["protocol"]["periods"]["retrospective_final"] == [
        "2026-01-01",
        "2026-08-04",
    ]
    assert payload["protocol"]["weather_availability_assumption"] == (
        "previous_day1 available before valid_time"
    )
    assert {item["kind"] for item in payload["limitations"]} == {
        "retrospective_final",
        "weather_run_provenance",
        "adaptive_interval_scope",
    }


def test_study_evidence_is_bounded_role_correct_hashed_and_not_duplicated(
    tmp_path, monkeypatch
):
    _, staging, _ = _release_tree(tmp_path)

    _run(staging / "config.yaml", monkeypatch)

    payload = json.loads((staging / "dashboard/src/generated/release.json").read_text())
    study = payload["study"]
    assert len(study["load_patterns"]["weekday"]) == 7
    assert len(study["load_patterns"]["month"]) == 12
    assert len(study["load_patterns"]["temperature"]) == 9
    assert len(study["load_patterns"]["shap_importance"]) == 3
    assert len(study["performance"]["daily_comparison"]) == 7
    assert len(study["performance"]["rolling_origin_mape"]) == 24
    assert len(study["performance"]["generalization_gap"]) == 7
    assert len(study["performance"]["hourly_ablation"]) == 12
    assert len(study["reliability"]["error_slices"]) == 210
    assert len(study["reliability"]["feature_drift"]) == 6
    assert len(study["reliability"]["residual_drift_monthly"]) == 8
    assert set(study["method"]["evidence_roles"].values()) == {
        "validation",
        "retrospective_final",
        "retrospective_explanation",
        "full_available_history_descriptive",
    }
    assert all(row["evaluation_period"] != "test" for row in study["reliability"]["error_slices"])
    assert all(
        "test_MAPE" not in row for row in study["performance"]["generalization_gap"]
    )
    assert len(payload["hourly_error"]["horizon_profile"]) == 3 * 24
    assert {row["slice_type"] for row in payload["intervals"]["hourly"]} == {
        "aggregate",
        "month",
        "horizon",
    }
    assert not {"intervals", "reconciliation", "comparison"} & set(study)

    hashes = payload["source_artifact_hashes"]
    for relative in (
        "analysis/load_profile_weekday.csv",
        "analysis/load_profile_month.csv",
        "analysis/temperature_load_curve.csv",
        "analysis/shap_importance.csv",
        "metrics/daily_comparison.csv",
        "metrics/rolling_origin_mape.csv",
        "metrics/generalization_gap.csv",
        "metrics/error_slices.csv",
        "hourly/coverage_by_month.csv",
        "hourly/coverage_by_horizon.csv",
        "hourly/residual_drift_monthly.csv",
    ):
        assert hashes[relative] == sha256_file(staging / "results" / relative)
    encoded = json.dumps(payload)
    assert "residual_drift.csv" not in encoded
    assert "coverage_by_local_day.csv" not in encoded
    assert ".png" not in encoded


@pytest.mark.parametrize("mutation", ["missing", "extra", "non_finite", "duplicate", "scope"])
def test_malformed_study_source_keeps_release_and_summary_unchanged(
    tmp_path, monkeypatch, mutation
):
    _, staging, records = _release_tree(tmp_path)
    destination = staging / "dashboard/src/generated/release.json"
    summary_path = staging / "results/run_summary.json"
    _run(staging / "config.yaml", monkeypatch)
    release_before = destination.read_bytes()
    path = staging / "results/analysis/load_profile_weekday.csv"
    frame = pd.read_csv(path)
    if mutation == "missing":
        path.unlink()
    elif mutation == "extra":
        frame["extra"] = "forbidden"
        frame.to_csv(path, index=False)
    elif mutation == "non_finite":
        frame.loc[0, "mean_load_MW"] = float("nan")
        frame.to_csv(path, index=False)
    elif mutation == "duplicate":
        pd.concat([frame, frame.iloc[[0]]], ignore_index=True).to_csv(path, index=False)
    else:
        frame["evidence_scope"] = "prospective"
        frame.to_csv(path, index=False)
    _refresh_identity(staging, records)
    summary_before = summary_path.read_bytes()

    with pytest.raises((AssertionError, ValueError)):
        _run(staging / "config.yaml", monkeypatch)

    assert destination.read_bytes() == release_before
    assert summary_path.read_bytes() == summary_before


def test_emit_frontend_release_contract(tmp_path, monkeypatch):
    output_value = os.environ.get("DASHBOARD_CONTRACT_OUT")
    if output_value is None:
        pytest.skip("DASHBOARD_CONTRACT_OUT is only set by the producer-consumer gate")
    output = Path(output_value)
    assert output.is_absolute()
    assert output.resolve().is_relative_to(ROOT)
    assert output.name == "05-04-dashboard-release-contract.json"
    _, staging, _ = _release_tree(tmp_path)

    _run(staging / "config.yaml", monkeypatch)
    payload = json.loads((staging / "dashboard/src/generated/release.json").read_text())
    assert payload["selector_options"]["hourly_point_error_horizons"] == list(range(1, 25))
    assert payload["selector_options"]["hourly_interval_coverage_horizons"] == [24]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


@pytest.mark.parametrize("mutation", ["missing", "non_finite", "mixed_identity", "wrong_role"])
def test_malformed_hourly_evidence_keeps_previous_dashboard(
    tmp_path, monkeypatch, mutation
):
    _, staging, records = _release_tree(tmp_path)
    destination = staging / "dashboard/src/generated/release.json"
    _run(staging / "config.yaml", monkeypatch)
    before = destination.read_bytes()
    comparison_path = staging / "results/hourly/hourly_comparison.csv"
    comparison = pd.read_csv(comparison_path)
    if mutation == "missing":
        comparison_path.unlink()
    elif mutation == "non_finite":
        comparison.loc[0, "MAE"] = float("nan")
        comparison.to_csv(comparison_path, index=False)
    elif mutation == "mixed_identity":
        comparison.loc[0, "protocol_fingerprint"] = "0" * 64
        comparison.to_csv(comparison_path, index=False)
    else:
        comparison.loc[0, "evaluation_period"] = "prospective_final"
        comparison.to_csv(comparison_path, index=False)
    _refresh_identity(staging, records)

    with pytest.raises((AssertionError, ValueError)):
        _run(staging / "config.yaml", monkeypatch)

    assert destination.read_bytes() == before


def test_incomplete_representative_week_keeps_previous_dashboard(tmp_path, monkeypatch):
    _, staging, _ = _release_tree(tmp_path)
    destination = staging / "dashboard/src/generated/release.json"
    _run(staging / "config.yaml", monkeypatch)
    before = destination.read_bytes()
    for state in ("test_predictions", "test_reconciled_predictions"):
        for model in ("residual_hybrid", "lightgbm_direct"):
            path = staging / f"results/hourly/{state}/{model}.csv"
            frame = pd.read_csv(path)
            frame.iloc[1:].to_csv(path, index=False)

    with pytest.raises(ValueError, match="complete seven-day"):
        build_dashboard_data._build_release_payload(staging / "results")

    assert destination.read_bytes() == before


def test_oversized_payload_keeps_previous_dashboard(tmp_path, monkeypatch):
    _, staging, _ = _release_tree(tmp_path)
    destination = staging / "dashboard/src/generated/release.json"
    _run(staging / "config.yaml", monkeypatch)
    before = destination.read_bytes()
    summary_path = staging / "results/run_summary.json"
    summary_before = summary_path.read_bytes()
    monkeypatch.setattr(build_dashboard_data, "_MAX_BYTES", 1)

    with pytest.raises(ValueError, match="exceeds"):
        _run(staging / "config.yaml", monkeypatch)

    assert destination.read_bytes() == before
    assert summary_path.read_bytes() == summary_before


@pytest.mark.parametrize("mutation", ["source_hash", "final_role"])
def test_invalid_source_keeps_previous_dashboard(tmp_path, monkeypatch, mutation):
    _, staging, _ = _release_tree(tmp_path)
    destination = staging / "dashboard/src/generated/release.json"
    _run(staging / "config.yaml", monkeypatch)
    before = destination.read_bytes()
    if mutation == "source_hash":
        with (staging / "results/predictions/ensemble.csv").open("a") as handle:
            handle.write("changed\n")
    else:
        path = staging / "results/run_summary.json"
        summary = json.loads(path.read_text())
        summary["final_role"] = "prospective_final"
        path.write_text(json.dumps(summary))

    with pytest.raises((AssertionError, ValueError)):
        _run(staging / "config.yaml", monkeypatch)

    assert destination.read_bytes() == before
