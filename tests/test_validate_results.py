from __future__ import annotations

import json

import pandas as pd
import pytest

from loadfc.presentation import (
    MANAGED_PRESENTATION_PATHS,
    IntervalEvidence,
    MetricEvidence,
    PresentationValues,
    render_managed_presentation,
)
from scripts import validate_results
from scripts.validate_results import (
    _check_hourly_predictions,
    _check_managed_presentations,
    _check_predictions,
    _check_reconciliation_artifacts,
    validate_protocol_contract,
)


def _presentation_values() -> PresentationValues:
    return PresentationValues(
        final_role="retrospective_final",
        final_role_label="Retrospective; previously inspected",
        final_start="2026-01-01",
        final_end="2026-08-04",
        source_revision="fixture-revision",
        bundle_fingerprint="b" * 64,
        daily_protocol_fingerprint="daily-fingerprint",
        hourly_protocol_fingerprint="hourly-fingerprint",
        daily=MetricEvidence("ensemble", 1003.6, 1307.3, 1.867, 0.445, 216),
        daily_baselines=(
            MetricEvidence("naive_1d", 3658.1, 5160.6, 7.025, 1.622, 216),
            MetricEvidence("seasonal_naive_7d", 2978.8, 4189.9, 5.575, 1.321, 216),
        ),
        hourly_model="residual_hybrid",
        daily_anchor_model="ensemble",
        hourly_validation_mae=1444.8,
        hourly_candidate_model="lightgbm_direct",
        hourly_candidate_validation_mae=1453.7,
        hourly_candidate_final_mae=1549.3,
        hourly_raw_mae=1627.6,
        hourly_reconciled_mae=1557.7,
        hourly_n=5183,
        bootstrap_difference=-8.4,
        bootstrap_lower=-20.0,
        bootstrap_upper=5.0,
        bootstrap_probability=0.8,
        bootstrap_n_days=216,
        hourly_intervals=(
            IntervalEvidence(
                "symmetric", "90%", 0.9, 0.8489, 5573.1, 8831.6, 5183,
                "empirical_retrospective",
            ),
            IntervalEvidence(
                "adaptive", "90%", 0.9, 0.8995, 6529.3, 8250.5, 5183,
                "prequential_monitoring_no_unconditional_time_series_coverage",
            ),
            IntervalEvidence(
                "cqr", "90%", 0.9, 0.8651, 7550.2, 10567.7, 5183,
                "empirical_retrospective",
            ),
        ),
    )


def _presentation_tree(root, values: PresentationValues) -> None:
    for relative in MANAGED_PRESENTATION_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"authored before\n{render_managed_presentation(values, relative)}\nauthored after\n"
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "value",
        "model",
        "role",
        "sample_size",
        "marker",
        "duplicate_marker",
        "prohibited_claim",
    ],
)
def test_managed_presentation_validator_rejects_drift_and_unqualified_claims(
    tmp_path, monkeypatch, mutation
):
    values = _presentation_values()
    _presentation_tree(tmp_path, values)
    readme = tmp_path / "README.md"
    text = readme.read_text()
    if mutation == "value":
        text = text.replace("MAPE 1.867%", "MAPE 9.999%")
    elif mutation == "model":
        text = text.replace("Daily forecast:** Ensemble", "Daily forecast:** LightGBM")
    elif mutation == "role":
        text = text.replace(
            "The final period was already examined",
            "The final period is a prospective holdout",
        )
    elif mutation == "sample_size":
        text = text.replace("n=216", "n=999", 1)
    elif mutation == "marker":
        text = text.replace("<!-- loadfc:generated-end -->", "")
    elif mutation == "duplicate_marker":
        text += "<!-- loadfc:generated-start -->\n<!-- loadfc:generated-end -->\n"
    else:
        text += "This is a production-ready forecasting service.\n"
    readme.write_text(text)
    monkeypatch.setattr(validate_results, "load_presentation_values", lambda _: values)

    with pytest.raises(ValueError):
        _check_managed_presentations(tmp_path, tmp_path / "results")


def _daily_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "actual": [100.0],
            "forecast": [99.0],
            "forecast_origin": ["2024-01-01"],
            "valid_time": ["2024-01-02"],
            "weather_source_run": ["open_meteo_previous_day1"],
            "weather_availability_assumption": ["previous_day1"],
        },
        index=pd.to_datetime(["2024-01-02"]),
    )


def _hourly_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "forecast_origin": ["2024-01-01T00:00:00+00:00", "2024-01-01T01:00:00+00:00"],
            "valid_time": ["2024-01-02T00:00:00+00:00", "2024-01-02T01:00:00+00:00"],
            "horizon": [24, 24],
            "actual": [100.0, 101.0],
            "prediction": [99.0, 100.0],
            "weather_source_run": ["persistence", "persistence"],
            "weather_availability_assumption": ["previous-day realized" for _ in range(2)],
        }
    )


def test_prediction_validator_accepts_a_complete_single_daily_row(tmp_path):
    path = tmp_path / "daily.csv"
    _daily_rows().to_csv(path)

    _check_predictions(path, pd.Timestamp("2024-01-02").date(), pd.Timestamp("2024-01-02").date(), 1)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("forecast_origin", "2024-01-02"),
        ("valid_time", "2024-01-03"),
    ],
)
def test_prediction_validator_rejects_impossible_daily_identity(tmp_path, column, value):
    path = tmp_path / "daily.csv"
    rows = _daily_rows()
    rows.loc[rows.index[0], column] = value
    rows.to_csv(path)

    with pytest.raises(AssertionError):
        _check_predictions(path, pd.Timestamp("2024-01-02").date(), pd.Timestamp("2024-01-02").date(), 1)


@pytest.mark.parametrize("mutation", ["empty", "missing", "null", "duplicate", "misordered", "oracle"])
def test_prediction_validator_rejects_ambiguous_daily_evidence(tmp_path, mutation):
    path = tmp_path / "daily.csv"
    rows = _daily_rows()
    if mutation == "empty":
        rows = rows.iloc[0:0]
    elif mutation == "missing":
        rows = rows.drop(columns="weather_source_run")
    elif mutation == "null":
        rows.loc[rows.index[0], "weather_availability_assumption"] = None
    elif mutation == "duplicate":
        rows = pd.concat([rows, rows])
    elif mutation == "misordered":
        later = rows.copy()
        later.index = pd.to_datetime(["2024-01-03"])
        rows = pd.concat([later, rows])
    else:
        rows.loc[rows.index[0], "weather_source_run"] = "oracle_sensitivity"
    rows.to_csv(path)

    with pytest.raises(AssertionError):
        _check_predictions(path, pd.Timestamp("2024-01-02").date(), pd.Timestamp("2024-01-03").date(), len(rows))


def test_hourly_validator_accepts_shared_metadata_for_distinct_utc_identities(tmp_path):
    path = tmp_path / "hourly.csv"
    _hourly_rows().to_csv(path, index=False)

    _check_hourly_predictions(path, required_columns=("actual", "prediction"))


def test_hourly_validator_rejects_valid_time_that_does_not_match_horizon(tmp_path):
    path = tmp_path / "hourly.csv"
    rows = _hourly_rows()
    rows.loc[0, "valid_time"] = "2024-01-01T12:00:00+00:00"
    rows.to_csv(path, index=False)

    with pytest.raises(AssertionError):
        _check_hourly_predictions(path, required_columns=("actual", "prediction"))


def test_hourly_validator_rejects_rows_outside_expected_period(tmp_path):
    path = tmp_path / "hourly.csv"
    _hourly_rows().to_csv(path, index=False)

    with pytest.raises(AssertionError):
        _check_hourly_predictions(
            path,
            required_columns=("actual", "prediction"),
            expected_start=pd.Timestamp("2024-01-03").date(),
            expected_end=pd.Timestamp("2024-01-03").date(),
        )


@pytest.mark.parametrize("mutation", ["empty", "missing", "null", "duplicate", "misordered", "oracle"])
def test_hourly_validator_rejects_ambiguous_evidence(tmp_path, mutation):
    path = tmp_path / "hourly.csv"
    rows = _hourly_rows()
    if mutation == "empty":
        rows = rows.iloc[0:0]
    elif mutation == "missing":
        rows = rows.drop(columns="weather_source_run")
    elif mutation == "null":
        rows.loc[0, "valid_time"] = None
    elif mutation == "duplicate":
        rows.loc[1, ["forecast_origin", "valid_time", "horizon"]] = rows.loc[
            0, ["forecast_origin", "valid_time", "horizon"]
        ]
    elif mutation == "misordered":
        rows = rows.iloc[::-1]
    else:
        rows.loc[0, "weather_source_run"] = "oracle_sensitivity"
    rows.to_csv(path, index=False)

    with pytest.raises(AssertionError):
        _check_hourly_predictions(path, required_columns=("actual", "prediction"))


_LEDGER = """# Evaluation Ledger

| Role | Inclusive period |
| --- | --- |
| train | 2019-01-14–2023-12-31 |
| validation | 2024-01-01–2025-06-30 |
| calibration | 2025-07-01–2025-12-31 |
| retrospective_final | 2026-01-01–2026-08-04 |

The current final role is `retrospective_final`, not a prospective holdout. The block has already
been inspected. Model and feature choices use training/validation evidence only.
"""


def _record(stream_id: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "source_revision": "abc123",
        "config_sha256": "config",
        "stream_id": stream_id,
        "model_identity": "test_model",
        "ordered_feature_columns": ["feature_a"],
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
        "interval_state_policy": {"fixed": "frozen"},
        "final_role": "retrospective_final",
        "validation_selection_evidence": "results/metrics/validation_metrics.csv",
        "rationale": "validation selects the model",
    }


def _metadata(record: dict[str, object], period: str) -> dict[str, object]:
    from loadfc.evaluation.protocol import protocol_fingerprint

    return {
        "stream_id": record["stream_id"],
        "protocol_fingerprint": protocol_fingerprint(record),
        "evaluation_period": period,
        "point_state_policy": json.dumps(record["point_state_policy"], sort_keys=True),
        "interval_state_policy": json.dumps(record["interval_state_policy"], sort_keys=True),
    }


def _protocol_tree(tmp_path):
    results = tmp_path / "results"
    ledger = tmp_path / "EVALUATION_LEDGER.md"
    ledger.write_text(_LEDGER)
    daily = _record("daily/test")
    hourly = _record("hourly/point/test")
    quantile = _record("hourly/quantile/test")
    records = {record["stream_id"]: record for record in (daily, hourly, quantile)}
    (results / "evaluation_protocol.json").parent.mkdir(parents=True)
    (results / "evaluation_protocol.json").write_text(json.dumps({"schema_version": 1, "records": records}))

    for directory, period in (("calibration_predictions", "calibration"), ("predictions", "retrospective_final")):
        path = results / directory
        path.mkdir()
        pd.DataFrame([_metadata(daily, period)]).to_csv(path / "test.csv", index=False)

    interval_dir = results / "interval_predictions"
    interval_dir.mkdir()
    pd.DataFrame(
        [
            {
                **_metadata(daily, "retrospective_final"),
                "actual": 1.0,
                "forecast": 1.0,
                "fixed_lower_80": 0.0,
                "fixed_upper_80": 2.0,
                "lower_80": 0.0,
                "upper_80": 2.0,
                "fixed_lower_95": 0.0,
                "fixed_upper_95": 2.0,
                "lower_95": 0.0,
                "upper_95": 2.0,
            }
        ]
    ).to_csv(interval_dir / "test.csv", index=False)
    metrics_dir = results / "metrics"
    metrics_dir.mkdir()
    pd.DataFrame(
        [
            {
                "model": "test",
                "method": method,
                "level": level,
                **_metadata(daily, "retrospective_final"),
                "coverage_scope": (
                    "empirical_retrospective"
                    if method == "fixed"
                    else "prequential_monitoring_no_unconditional_time_series_coverage"
                ),
                "nominal": nominal,
                "empirical_coverage": 1.0,
                "mean_width_MW": 2.0,
                "interval_score_MW": 2.0,
                "n": 1,
            }
            for method in ("fixed", "adaptive")
            for level, nominal in (("80%", 0.8), ("95%", 0.95))
        ]
    ).to_csv(metrics_dir / "interval_coverage.csv", index=False)

    hourly_dir = results / "hourly"
    hourly_dir.mkdir()
    final_valid_time = pd.date_range("2026-03-28T23:00Z", periods=23, freq="h")
    for name, record in (("calibration_predictions.csv", hourly), ("test_intervals.csv", hourly)):
        period = "calibration" if name.startswith("calibration") else "retrospective_final"
        valid_times = final_valid_time[:1] if period == "calibration" else final_valid_time
        rows = [
            {
                **_metadata(record, period),
                "forecast_origin": valid_time - pd.Timedelta(hours=24),
                "valid_time": valid_time,
                "horizon": 24,
                "actual": 1.0,
                "prediction": 1.0,
                "lower": 0.0,
                "upper": 2.0,
            }
            for valid_time in valid_times
        ]
        pd.DataFrame(rows).to_csv(hourly_dir / name, index=False)
    for name, record in (("cqr_calibration_predictions.csv", quantile), ("cqr_test_intervals.csv", quantile)):
        period = "calibration" if name.startswith("cqr_calibration") else "retrospective_final"
        bounds = (
            {"lower_quantile": 0.0, "upper_quantile": 2.0}
            if period == "calibration"
            else {"lower": 0.0, "upper": 2.0}
        )
        valid_times = final_valid_time[:1] if period == "calibration" else final_valid_time
        rows = [
            {
                **_metadata(record, period),
                "forecast_origin": valid_time - pd.Timedelta(hours=24),
                "valid_time": valid_time,
                "horizon": 24,
                "actual": 1.0,
                "prediction": 1.0,
                **bounds,
                "quantile_crossed": False,
            }
            for valid_time in valid_times
        ]
        pd.DataFrame(rows).to_csv(hourly_dir / name, index=False)

    evidence = []
    for method in ("symmetric", "adaptive", "cqr"):
        record = quantile if method == "cqr" else hourly
        scope = "prequential_monitoring_no_unconditional_time_series_coverage" if method == "adaptive" else "empirical_retrospective"
        for slice_type, values in (("aggregate", ["all"]), ("horizon", [24]), ("local_day", ["2026-03-29"])):
            for value in values:
                evidence.append(
                    {
                        "method": method,
                        "level": "90%",
                        "slice_type": slice_type,
                        "slice_value": value,
                        "evaluation_period": "retrospective_final",
                        "coverage_scope": scope,
                        "stream_id": record["stream_id"],
                        "protocol_fingerprint": _metadata(record, "retrospective_final")["protocol_fingerprint"],
                        "nominal": 0.9,
                        "empirical_coverage": 0.9,
                        "mean_width": 2.0,
                        "interval_score": 2.0,
                        "n": 23,
                    }
                )
    evidence_frame = pd.DataFrame(evidence)
    for name, slice_type in (("interval_comparison.csv", "aggregate"), ("coverage_by_horizon.csv", "horizon"), ("coverage_by_local_day.csv", "local_day")):
        evidence_frame[evidence_frame["slice_type"].eq(slice_type)].to_csv(hourly_dir / name, index=False)

    analysis_dir = results / "analysis"
    analysis_dir.mkdir()
    scope = "full_available_history_descriptive"
    pd.DataFrame(
        [
            {
                "weekday_order": order,
                "weekday": weekday,
                "mean_load_MW": 50_000.0 + order,
                "n_days": 52,
                "evidence_scope": scope,
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
        ]
    ).to_csv(analysis_dir / "load_profile_weekday.csv", index=False)
    pd.DataFrame(
        [
            {
                "month_order": order,
                "month": month,
                "mean_load_MW": 50_000.0 + order,
                "n_days": 30,
                "evidence_scope": scope,
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
        ]
    ).to_csv(analysis_dir / "load_profile_month.csv", index=False)
    pd.DataFrame(
        [
            {
                "bin_order": order,
                "lower_C": lower,
                "upper_C": lower + 5,
                "mean_load_MW": 50_000.0 + order,
                "n_days": 10,
                "evidence_scope": scope,
            }
            for order, lower in enumerate(range(-10, 35, 5), start=1)
        ]
    ).to_csv(analysis_dir / "temperature_load_curve.csv", index=False)

    months = [f"2026-{month:02d}" for month in range(1, 9)]
    monthly_rows = []
    for method in ("symmetric", "adaptive", "cqr"):
        record = quantile if method == "cqr" else hourly
        scope_value = (
            "prequential_monitoring_no_unconditional_time_series_coverage"
            if method == "adaptive"
            else "empirical_retrospective"
        )
        for month in months:
            monthly_rows.append(
                {
                    "method": method,
                    "level": "90%",
                    "slice_type": "month",
                    "slice_value": month,
                    "evaluation_period": "retrospective_final",
                    "coverage_scope": scope_value,
                    "stream_id": record["stream_id"],
                    "protocol_fingerprint": _metadata(record, "retrospective_final")[
                        "protocol_fingerprint"
                    ],
                    "nominal": 0.9,
                    "empirical_coverage": 0.9,
                    "mean_width": 2.0,
                    "interval_score": 2.0,
                    "n": 24,
                }
            )
    pd.DataFrame(monthly_rows).to_csv(hourly_dir / "coverage_by_month.csv", index=False)
    pd.DataFrame(
        [
            {
                "month": month,
                "evaluation_period": "retrospective_final",
                "monitoring_scope": "prequential_monitoring",
                "stream_id": hourly["stream_id"],
                "protocol_fingerprint": _metadata(hourly, "retrospective_final")[
                    "protocol_fingerprint"
                ],
                "mean_absolute_error_MW": 1.0,
                "max_page_hinkley_statistic": 2.0,
                "alert_count": 0,
                "n": 1,
            }
            for month in months
        ]
    ).to_csv(hourly_dir / "residual_drift_monthly.csv", index=False)
    pd.DataFrame(
        [
            {"value": 1.0, "running_mean": 1.0, "statistic": 2.0, "alert": False}
            for _ in months
        ]
    ).to_csv(hourly_dir / "residual_drift.csv", index=False)

    fingerprints = {name: _metadata(record, "retrospective_final")["protocol_fingerprint"] for name, record in records.items()}
    (results / "run_summary.json").write_text(
        json.dumps(
            {
                "final_role": "retrospective_final",
                "protocol_manifest": "evaluation_protocol.json",
                "protocol_schema_version": 1,
                "protocol_fingerprints": fingerprints,
                "retrospective_final_metrics": {"test": {"MAE": 1.0}},
                "test_metrics": {"test": {"MAE": 1.0}},
            }
        )
    )
    _write_comparison_evidence(results)
    return results, ledger


def _write_comparison_evidence(results):
    manifest_path = results / "evaluation_protocol.json"
    manifest = json.loads(manifest_path.read_text())
    records = manifest["records"]
    other = _record("hourly/point/other")
    records[other["stream_id"]] = other
    manifest_path.write_text(json.dumps(manifest))

    summary_path = results / "run_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["protocol_fingerprints"] = {
        stream_id: _metadata(record, "retrospective_final")["protocol_fingerprint"]
        for stream_id, record in records.items()
    }
    summary_path.write_text(json.dumps(summary))

    daily = records["daily/test"]
    pd.DataFrame(
        [
            {
                "model": "test",
                "evaluation_period": "retrospective_final",
                "stream_id": "daily/test",
                "protocol_fingerprint": _metadata(daily, "retrospective_final")[
                    "protocol_fingerprint"
                ],
                "dates": "2026-01-01:2026-01-01",
                "start": "2026-01-01",
                "end": "2026-01-01",
                "n": 1,
                "MAE": 1.0,
                "RMSE": 1.0,
                "MAPE": 1.0,
                "MASE": 1.0,
            }
        ]
    ).to_csv(results / "metrics/daily_comparison.csv", index=False)

    hourly = results / "hourly"
    predictions = hourly / "test_predictions"
    predictions.mkdir()
    valid_time = pd.date_range("2026-01-01T23:00Z", periods=24, freq="h")
    for name, record in (("test", records["hourly/point/test"]), ("other", other)):
        pd.DataFrame(
            {
                "forecast_origin": valid_time - pd.Timedelta(hours=24),
                "valid_time": valid_time,
                "horizon": 24,
                "actual": 100.0,
                "prediction": 99.0 if name == "test" else 98.0,
                **_metadata(record, "retrospective_final"),
            }
        ).to_csv(predictions / f"{name}.csv", index=False)

    selection_evidence = "results/hourly/model_ablation_validation_reconciled.csv"
    pd.DataFrame([{"model": "test", "hourly_MAE": 1.0}]).to_csv(
        hourly / "model_ablation_validation_reconciled.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "selected_model": "test",
                "daily_anchor_model": "test",
                "selection_metric": "reconciled_h24_hourly_MAE",
                "selection_period_start": "2024-01-01",
                "selection_period_end": "2025-06-30",
                "selection_evidence_period": "validation",
                "selection_protocol_fingerprint": _metadata(
                    records["hourly/point/test"], "retrospective_final"
                )["protocol_fingerprint"],
            }
        ]
    ).to_csv(hourly / "model_selection.csv", index=False)
    comparison_rows = []
    for name, record in (("test", records["hourly/point/test"]), ("other", other)):
        comparison_rows.append(
            {
                "model": name,
                "slice_type": "aggregate",
                "slice_value": "all",
                "evaluation_period": "retrospective_final",
                "stream_id": record["stream_id"],
                "protocol_fingerprint": _metadata(record, "retrospective_final")[
                    "protocol_fingerprint"
                ],
                "eligible_horizon": "24",
                "local_start": valid_time.min().isoformat(),
                "local_end": valid_time.max().isoformat(),
                "n": 24,
                "MAE": 1.0,
                "RMSE": 1.0,
                "MAPE": 1.0,
                "MASE": 1.0,
                "validation_selection_metric": "reconciled_h24_hourly_MAE",
                "validation_selected_model": "test",
                "validation_selection_evidence": selection_evidence,
            }
        )
    pd.DataFrame(comparison_rows).to_csv(hourly / "hourly_comparison.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate": "other",
                "reference": "test",
                "mae_difference": 1.0,
                "ci_lower": -1.0,
                "ci_upper": 2.0,
                "probability_candidate_better": 0.25,
                "n_days": 1,
                "seed": 42,
                "n_bootstrap": 100,
                "block_size_days": 7,
                "practical_tie_threshold": 50.0,
                "practical_tie": True,
                "selected_model": "test",
                "selected_model_preserved": True,
                "uncertainty_scope": "retrospective_final realized-weather time-series block bootstrap",
                "validation_selection_metric": "reconciled_h24_hourly_MAE",
                "validation_selected_model": "test",
                "validation_selection_evidence": selection_evidence,
            }
        ]
    ).to_csv(hourly / "model_ablation_test_uncertainty.csv", index=False)


def test_protocol_validator_rejects_inexact_hourly_aggregate_n(tmp_path):
    results, ledger = _protocol_tree(tmp_path)
    path = results / "hourly/hourly_comparison.csv"
    frame = pd.read_csv(path)
    frame.loc[frame["slice_type"].eq("aggregate"), "n"] = 999
    frame.to_csv(path, index=False)

    with pytest.raises((AssertionError, ValueError)):
        validate_protocol_contract(results, ledger)


@pytest.mark.parametrize(
    ("relative_path", "invalid_n"),
    [
        ("hourly/interval_comparison.csv", 999),
        ("hourly/interval_comparison.csv", 22),
        ("hourly/interval_comparison.csv", 24),
        ("hourly/coverage_by_horizon.csv", 999),
        ("hourly/coverage_by_horizon.csv", 24),
        ("hourly/coverage_by_local_day.csv", 24),
    ],
)
def test_protocol_validator_rejects_inexact_hourly_interval_evidence_n(
    tmp_path, relative_path, invalid_n
):
    results, ledger = _protocol_tree(tmp_path)
    path = results / relative_path
    frame = pd.read_csv(path)
    frame["n"] = invalid_n
    frame.to_csv(path, index=False)

    with pytest.raises((AssertionError, ValueError)):
        validate_protocol_contract(results, ledger)


@pytest.mark.parametrize(
    "mutation",
    [
        "daily_missing_column",
        "daily_empty",
        "daily_non_finite",
        "daily_wrong_role",
        "hourly_wrong_fingerprint",
        "hourly_cross_period",
        "hourly_invalid_selection",
        "bootstrap_missing_column",
        "bootstrap_non_finite",
        "bootstrap_wrong_seed",
    ],
)
def test_protocol_validator_rejects_invalid_comparison_and_bootstrap_artifacts(
    tmp_path, mutation
):
    results, ledger = _protocol_tree(tmp_path)
    if mutation.startswith("daily"):
        path = results / "metrics/daily_comparison.csv"
    elif mutation.startswith("hourly"):
        path = results / "hourly/hourly_comparison.csv"
    else:
        path = results / "hourly/model_ablation_test_uncertainty.csv"
    frame = pd.read_csv(path)
    if mutation == "daily_missing_column":
        frame = frame.drop(columns="MAE")
    elif mutation == "daily_empty":
        frame = frame.iloc[0:0]
    elif mutation == "daily_non_finite":
        frame.loc[0, "MAE"] = float("inf")
    elif mutation == "daily_wrong_role":
        frame.loc[0, "evaluation_period"] = "calibration"
    elif mutation == "hourly_wrong_fingerprint":
        frame.loc[0, "protocol_fingerprint"] = "wrong"
    elif mutation == "hourly_cross_period":
        frame.loc[0, "evaluation_period"] = "calibration"
    elif mutation == "hourly_invalid_selection":
        frame.loc[0, "validation_selected_model"] = "other"
    elif mutation == "bootstrap_missing_column":
        frame = frame.drop(columns="n_bootstrap")
    elif mutation == "bootstrap_non_finite":
        frame.loc[0, "ci_lower"] = float("inf")
    else:
        frame.loc[0, "seed"] = 7
    frame.to_csv(path, index=False)

    with pytest.raises((AssertionError, ValueError)):
        validate_protocol_contract(results, ledger)


def test_protocol_contract_accepts_complete_synthetic_tree(tmp_path):
    results, ledger = _protocol_tree(tmp_path)

    validate_protocol_contract(results, ledger)


@pytest.mark.parametrize(
    "mutation",
    [
        "weekday_order",
        "temperature_bin",
        "missing_method_month",
        "residual_count",
        "residual_identity",
    ],
)
def test_protocol_contract_rejects_malformed_study_evidence(tmp_path, mutation):
    results, ledger = _protocol_tree(tmp_path)
    if mutation == "weekday_order":
        path = results / "analysis/load_profile_weekday.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "weekday_order"] = 2
    elif mutation == "temperature_bin":
        path = results / "analysis/temperature_load_curve.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "lower_C"] = -11
    elif mutation == "missing_method_month":
        path = results / "hourly/coverage_by_month.csv"
        frame = pd.read_csv(path).iloc[1:]
    elif mutation == "residual_count":
        path = results / "hourly/residual_drift_monthly.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "n"] = 2
    else:
        path = results / "hourly/residual_drift_monthly.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "protocol_fingerprint"] = "wrong"
    frame.to_csv(path, index=False)

    with pytest.raises((AssertionError, ValueError)):
        validate_results.validate_protocol_contract(results, ledger)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("protocol_fingerprint", "wrong"),
        ("evaluation_period", "prospective_final"),
        ("point_state_policy", '{"fit_through":"other"}'),
        ("interval_state_policy", '{"fixed":"other"}'),
        ("stream_id", "daily/unregistered"),
    ],
)
def test_protocol_contract_rejects_daily_interval_identity_mutation(tmp_path, column, value):
    results, ledger = _protocol_tree(tmp_path)
    path = results / "interval_predictions" / "test.csv"
    frame = pd.read_csv(path)
    frame[column] = value
    frame.to_csv(path, index=False)

    with pytest.raises((AssertionError, ValueError)):
        validate_protocol_contract(results, ledger)


@pytest.mark.parametrize("mutation", ["missing_predictions", "missing_summary", "summary_disagrees"])
def test_protocol_contract_rejects_incomplete_daily_interval_evidence(tmp_path, mutation):
    results, ledger = _protocol_tree(tmp_path)
    if mutation == "missing_predictions":
        (results / "interval_predictions" / "test.csv").unlink()
    elif mutation == "missing_summary":
        (results / "metrics" / "interval_coverage.csv").unlink()
    else:
        path = results / "metrics" / "interval_coverage.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "empirical_coverage"] = 0.0
        frame.to_csv(path, index=False)

    with pytest.raises((AssertionError, ValueError)):
        validate_protocol_contract(results, ledger)


def test_reconciliation_validator_rejects_failed_invariant(tmp_path):
    hourly = tmp_path / "hourly"
    hourly.mkdir()
    pd.DataFrame(
        [{
            "local_date": "2026-01-01",
            "reconciled_hourly_mean": 100.0,
            "daily_anchor": 100.0,
            "delta": 0.1,
            "abs_delta": 0.1,
            "tolerance": 0.01,
            "pass": False,
            "n": 24,
        }]
    ).to_csv(hourly / "reconciliation_invariants.csv", index=False)
    pd.DataFrame(
        [{
            "model": "raw",
            "evaluation_period": "retrospective_final",
            "stream_id": "hourly/point/test",
            "protocol_fingerprint": "wrong",
            "MAE": 1.0,
            "n_hours": 24,
        }]
    ).to_csv(hourly / "reconciliation_metrics.csv", index=False)
    with pytest.raises(AssertionError):
        _check_reconciliation_artifacts(hourly, {"hourly/point/test": {}})


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_ledger",
        "missing_manifest",
        "wrong_final_role",
        "post_validation_selection",
        "fingerprint_mismatch",
        "missing_interval_score",
        "sparse_group",
        "missing_horizon",
        "quantile_crossing",
        "summary_disagreement",
    ],
)
def test_protocol_contract_rejects_each_declared_mutation(tmp_path, mutation):
    results, ledger = _protocol_tree(tmp_path)
    hourly = results / "hourly"
    if mutation == "missing_ledger":
        ledger.unlink()
    elif mutation == "missing_manifest":
        (results / "evaluation_protocol.json").unlink()
    elif mutation == "wrong_final_role":
        path = results / "predictions" / "test.csv"
        frame = pd.read_csv(path)
        frame["evaluation_period"] = "prospective_final"
        frame.to_csv(path, index=False)
    elif mutation == "post_validation_selection":
        path = results / "evaluation_protocol.json"
        payload = json.loads(path.read_text())
        payload["records"]["daily/test"]["validation_selection_evidence"] = "results/metrics/test_metrics.csv"
        path.write_text(json.dumps(payload))
    elif mutation == "fingerprint_mismatch":
        path = results / "hourly" / "test_intervals.csv"
        frame = pd.read_csv(path)
        frame["protocol_fingerprint"] = "wrong"
        frame.to_csv(path, index=False)
    elif mutation == "missing_interval_score":
        path = hourly / "interval_comparison.csv"
        frame = pd.read_csv(path).drop(columns="interval_score")
        frame.to_csv(path, index=False)
    elif mutation == "sparse_group":
        path = hourly / "coverage_by_local_day.csv"
        frame = pd.read_csv(path)
        frame["n"] = 22
        frame.to_csv(path, index=False)
    elif mutation == "missing_horizon":
        path = hourly / "coverage_by_horizon.csv"
        frame = pd.read_csv(path).query("slice_value != 24")
        frame.to_csv(path, index=False)
    elif mutation == "quantile_crossing":
        path = hourly / "cqr_test_intervals.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "quantile_crossed"] = True
        frame.to_csv(path, index=False)
    else:
        path = results / "run_summary.json"
        payload = json.loads(path.read_text())
        payload["final_role"] = "prospective_final"
        path.write_text(json.dumps(payload))

    with pytest.raises((AssertionError, ValueError)):
        validate_protocol_contract(results, ledger)
