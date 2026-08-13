"""Check generated result files for consistency with the data and run summary."""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import math
import subprocess
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from loadfc import __version__
from loadfc.config import Config
from loadfc.evaluation.conformal import interval_evidence
from loadfc.evaluation.protocol import (
    assert_compatible_artifacts,
    protocol_fingerprint,
    validate_protocol_record,
)
from loadfc.presentation import (
    MANAGED_PRESENTATION_PATHS,
    check_managed_presentation,
    load_presentation_values,
)
from loadfc.tracking import (
    PRESENTATION_ARTIFACTS,
    bundle_fingerprint,
    git_commit,
    sha256_file,
    source_is_clean,
)

MODELS = {
    "SARIMAX",
    "xgboost",
    "lightgbm",
    "random_forest",
    "naive_1d",
    "seasonal_naive_7d",
    "ensemble",
}
_PROVENANCE_COLUMNS = [
    "forecast_origin",
    "valid_time",
    "weather_source_run",
    "weather_availability_assumption",
]
_HOURLY_IDENTITY = ["forecast_origin", "valid_time", "horizon"]
_PROTOCOL_COLUMNS = [
    "stream_id",
    "protocol_fingerprint",
    "evaluation_period",
    "point_state_policy",
    "interval_state_policy",
]
_HOURLY_EVIDENCE_COLUMN_ORDER = [
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
_HOURLY_EVIDENCE_COLUMNS = set(_HOURLY_EVIDENCE_COLUMN_ORDER)
_DAILY_EVIDENCE_COLUMNS = {
    "model",
    "method",
    "level",
    "evaluation_period",
    "stream_id",
    "protocol_fingerprint",
    "point_state_policy",
    "interval_state_policy",
    "coverage_scope",
    "nominal",
    "empirical_coverage",
    "mean_width_MW",
    "interval_score_MW",
    "n",
}
_MIN_HOURLY_GROUP_N = 23
_RECONCILIATION_INVARIANT_COLUMNS = {
    "local_date",
    "reconciled_hourly_mean",
    "daily_anchor",
    "delta",
    "abs_delta",
    "tolerance",
    "pass",
    "n",
}
_RECONCILIATION_METADATA = {"evaluation_period", "stream_id", "protocol_fingerprint"}
_PRESENTATION_ARTIFACTS = set(PRESENTATION_ARTIFACTS)
_RESULT_SUFFIXES = {".csv", ".json", ".png"}
_DESCRIPTIVE_SCOPE = "full_available_history_descriptive"
_WEEKDAY_COLUMNS = [
    "weekday_order",
    "weekday",
    "mean_load_MW",
    "n_days",
    "evidence_scope",
]
_MONTH_COLUMNS = [
    "month_order",
    "month",
    "mean_load_MW",
    "n_days",
    "evidence_scope",
]
_TEMPERATURE_COLUMNS = [
    "bin_order",
    "lower_C",
    "upper_C",
    "mean_load_MW",
    "n_days",
    "evidence_scope",
]
_MONTHLY_RESIDUAL_COLUMNS = [
    "month",
    "evaluation_period",
    "monitoring_scope",
    "stream_id",
    "protocol_fingerprint",
    "mean_absolute_error_MW",
    "max_page_hinkley_statistic",
    "alert_count",
    "n",
]
_DAILY_COMPARISON_COLUMNS = {
    "model",
    "evaluation_period",
    "stream_id",
    "protocol_fingerprint",
    "dates",
    "start",
    "end",
    "n",
    "MAE",
    "RMSE",
    "MAPE",
    "MASE",
}
_SELECTION_COLUMNS = {
    "validation_selection_metric",
    "validation_selected_model",
    "validation_selection_evidence",
}
_HOURLY_COMPARISON_COLUMNS = {
    "model",
    "slice_type",
    "slice_value",
    "evaluation_period",
    "stream_id",
    "protocol_fingerprint",
    "eligible_horizon",
    "local_start",
    "local_end",
    "n",
    "MAE",
    "RMSE",
    "MAPE",
    "MASE",
    *_SELECTION_COLUMNS,
}
_BOOTSTRAP_COLUMNS = {
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
    *_SELECTION_COLUMNS,
}


def _require_assertions_enabled() -> None:
    """Keep this fail-closed validator from running with checks optimized away."""

    if not __debug__:
        raise RuntimeError("validate_results.py must not run with python -O")


def _source_is_clean(source_root: Path, artifact_root: Path) -> bool:
    return source_is_clean(
        source_root,
        artifact_root,
        allowed_paths=(*MANAGED_PRESENTATION_PATHS, *_PRESENTATION_ARTIFACTS),
    )


def _canonical_result_hashes(results: Path) -> dict[str, str]:
    return {
        path.relative_to(results).as_posix(): sha256_file(path)
        for path in sorted(results.rglob("*"))
        if path.is_file()
        and path.name != "run_summary.json"
        and path.suffix in _RESULT_SUFFIXES
        and not any(
            part.startswith(".") or part == "__pycache__"
            for part in path.relative_to(results).parts
        )
    }


def _validate_release_manifest(
    summary: dict[str, object],
    results: Path,
    records: dict[str, dict[str, object]],
    source_root: Path,
    config_path: Path,
) -> None:
    source_root = source_root.resolve()
    config_path = config_path.resolve()
    artifact_root = config_path.parent
    expected_fingerprints = {
        stream_id: protocol_fingerprint(record) for stream_id, record in records.items()
    }
    assert summary["source_revision"] == git_commit(source_root), "source revision conflict"
    assert summary["source_clean"] is True, "release source was not clean"
    assert _source_is_clean(source_root, artifact_root), "current source is not clean"
    assert summary["config_sha256"] == sha256_file(config_path), "config hash conflict"
    assert summary["lock_sha256"] == sha256_file(source_root / "uv.lock"), "lock hash conflict"
    assert summary["protocol_fingerprints"] == expected_fingerprints, "protocol hash conflict"
    for record in records.values():
        assert record["source_revision"] == summary["source_revision"], "protocol source conflict"
        assert record["config_sha256"] == summary["config_sha256"], "protocol config conflict"

    artifact_hashes = _canonical_result_hashes(results)
    assert summary["artifact_hashes"] == artifact_hashes, "result artifact hash conflict"
    declared = summary["declared_artifacts"]
    assert isinstance(declared, dict), "declared_artifacts must be a mapping"
    expected_keys = {f"results/{relative}" for relative in artifact_hashes} | _PRESENTATION_ARTIFACTS
    assert set(declared) == expected_keys, "declared artifact set conflict"
    for relative, digest in artifact_hashes.items():
        assert declared[f"results/{relative}"] == digest, f"declared result hash conflict: {relative}"
    for relative in _PRESENTATION_ARTIFACTS:
        path = artifact_root / relative
        digest = declared[relative]
        if digest is None:
            assert not path.exists(), f"undeclared presentation artifact exists: {relative}"
        else:
            assert path.is_file(), f"missing declared artifact: {relative}"
            assert digest == sha256_file(path), f"declared artifact hash conflict: {relative}"
    assert isinstance(declared["report/generated_results.tex"], str), "report input is undeclared"

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
    expected_bundle = bundle_fingerprint(fingerprint_fields)
    assert summary["bundle_fingerprint"] == expected_bundle, "bundle fingerprint conflict"

    dashboard_path = artifact_root / "dashboard/src/generated/release.json"
    if dashboard_path.is_file():
        dashboard = json.loads(dashboard_path.read_text())
        assert dashboard["schema_version"] == 1, "dashboard schema conflict"
        for field in ("source_revision", "bundle_fingerprint", "final_role"):
            assert dashboard[field] == summary[field], f"dashboard {field} conflict"
        assert dashboard["generated_at"] == summary["generated_at_utc"], (
            "dashboard generated timestamp conflict"
        )
        source_hashes = dashboard["source_artifact_hashes"]
        assert isinstance(source_hashes, dict) and source_hashes, "dashboard sources missing"
        for relative, digest in source_hashes.items():
            assert artifact_hashes.get(relative) == digest, (
                f"dashboard source hash conflict: {relative}"
            )


def _check_predictions(
    path: Path,
    start,
    end,
    expected_rows: int,
    *,
    required_columns: tuple[str, ...] = ("actual", "forecast"),
) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0, parse_dates=[0])
    assert not frame.empty, path
    assert set([*required_columns, *_PROVENANCE_COLUMNS]) <= set(frame), path
    assert len(frame) == expected_rows, path
    assert frame.index.is_unique and frame.index.is_monotonic_increasing, path
    assert frame.index.min().date() == start, path
    assert frame.index.max().date() == end, path
    assert frame[[*required_columns, *_PROVENANCE_COLUMNS]].notna().all().all(), path
    assert not frame["weather_source_run"].eq("oracle_sensitivity").any(), path
    valid_time = pd.to_datetime(frame["valid_time"], errors="coerce")
    forecast_origin = pd.to_datetime(frame["forecast_origin"], errors="coerce")
    assert valid_time.notna().all() and forecast_origin.notna().all(), path
    assert (valid_time.dt.normalize().to_numpy() == frame.index.normalize().to_numpy()).all(), path
    assert (valid_time - forecast_origin).eq(pd.Timedelta(days=1)).all(), path
    for column in required_columns:
        assert pd.to_numeric(frame[column], errors="coerce").notna().all(), path
    return frame


def _check_hourly_predictions(
    path: Path,
    *,
    required_columns: tuple[str, ...],
    expected_start=None,
    expected_end=None,
) -> pd.DataFrame:
    """Validate flat, UTC-identified hourly evidence before presentation consumes it."""

    frame = pd.read_csv(path)
    required = {*_HOURLY_IDENTITY, *_PROVENANCE_COLUMNS, *required_columns}
    assert not frame.empty and required <= set(frame), path
    assert frame[list(required)].notna().all().all(), path
    parsed: dict[str, pd.Series] = {}
    for column in ("forecast_origin", "valid_time"):
        raw = frame[column].astype(str)
        assert raw.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$").all(), path
        parsed[column] = pd.to_datetime(raw, errors="coerce", utc=True)
        assert parsed[column].notna().all(), path
    if expected_start is not None or expected_end is not None:
        local_dates = parsed["valid_time"].dt.tz_convert("Europe/Berlin").dt.date
        if expected_start is not None:
            assert (local_dates >= expected_start).all(), path
        if expected_end is not None:
            assert (local_dates <= expected_end).all(), path
    horizon = pd.to_numeric(frame["horizon"], errors="coerce")
    assert horizon.notna().all() and (horizon > 0).all() and (horizon % 1 == 0).all(), path
    for column in required_columns:
        assert pd.to_numeric(frame[column], errors="coerce").notna().all(), path
    identity = pd.DataFrame(
        {
            "forecast_origin": parsed["forecast_origin"],
            "valid_time": parsed["valid_time"],
            "horizon": horizon.astype(int),
        }
    )
    assert (identity["valid_time"] - identity["forecast_origin"]).eq(
        pd.to_timedelta(identity["horizon"], unit="h")
    ).all(), path
    assert not identity.duplicated().any(), path
    assert identity.equals(identity.sort_values(_HOURLY_IDENTITY, kind="stable").reset_index(drop=True)), path
    assert not frame["weather_source_run"].eq("oracle_sensitivity").any(), path
    return frame


def _check_hourly_candidates(
    hourly: Path,
    *,
    metrics_name: str,
    artifacts_name: str,
) -> None:
    models = set(pd.read_csv(hourly / metrics_name, index_col=0).index.astype(str))
    artifacts = hourly / artifacts_name
    files = {path.stem: path for path in artifacts.glob("*.csv")}
    assert files.keys() == models, artifacts
    for path in files.values():
        _check_hourly_predictions(path, required_columns=("actual", "prediction"))


def _check_ledger(path: Path) -> None:
    """Require the human-readable frozen chronology before consuming artifacts."""

    assert path.is_file(), f"missing evaluation ledger: {path}"
    text = path.read_text(encoding="utf-8")
    for required in (
        "2019-01-14–2023-12-31",
        "2024-01-01–2025-06-30",
        "2025-07-01–2025-12-31",
        "2026-01-01–2026-08-04",
        "retrospective_final",
        "inspected",
        "training/validation evidence only",
    ):
        assert required in text, f"ledger missing required protocol statement: {required}"


def _load_protocol_manifest(results: Path) -> dict[str, dict[str, object]]:
    path = results / "evaluation_protocol.json"
    assert path.is_file(), f"missing protocol manifest: {path}"
    payload = json.loads(path.read_text())
    assert payload.get("schema_version") == 1, f"invalid protocol manifest schema: {path}"
    records = payload.get("records")
    assert isinstance(records, dict) and records, f"protocol manifest has no records: {path}"
    normalized: dict[str, dict[str, object]] = {}
    for stream_id, record in records.items():
        assert isinstance(record, dict), f"protocol record is not a mapping: {stream_id}"
        checked = validate_protocol_record(record)
        assert stream_id == checked["stream_id"], f"protocol stream key mismatch: {stream_id}"
        assert checked["final_role"] == "retrospective_final", stream_id
        evidence = str(checked["validation_selection_evidence"])
        assert "validation" in evidence and not any(
            marker in evidence.lower() for marker in ("calibration", "test", "final")
        ), f"post-validation selection evidence: {stream_id}"
        normalized[stream_id] = checked
    return normalized


def _metadata_from_frame(frame: pd.DataFrame, path: Path, period: str) -> dict[str, object]:
    assert set(_PROTOCOL_COLUMNS) <= set(frame), f"missing protocol metadata: {path}"
    metadata: dict[str, object] = {}
    for column in _PROTOCOL_COLUMNS:
        values = frame[column].dropna().unique()
        assert len(values) == 1, f"artifact must have one {column}: {path}"
        metadata[column] = values[0]
    assert metadata["evaluation_period"] == period, f"invalid period: {path}"
    return metadata


def _check_registered_pair(
    calibration_path: Path,
    final_path: Path,
    records: dict[str, dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    assert calibration_path.is_file(), f"missing calibration artifact: {calibration_path}"
    assert final_path.is_file(), f"missing final artifact: {final_path}"
    calibration = pd.read_csv(calibration_path)
    final = pd.read_csv(final_path)
    calibration_metadata = _metadata_from_frame(calibration, calibration_path, "calibration")
    final_metadata = _metadata_from_frame(final, final_path, "retrospective_final")
    assert_compatible_artifacts(calibration_metadata, final_metadata)
    stream_id = str(final_metadata["stream_id"])
    assert stream_id in records, f"unregistered stream {stream_id}: {final_path}"
    record = records[stream_id]
    assert final_metadata["protocol_fingerprint"] == protocol_fingerprint(record), final_path
    for field in ("point_state_policy", "interval_state_policy"):
        assert final_metadata[field] == json.dumps(record[field], sort_keys=True), final_path
    return calibration, final, final_metadata


def _check_cqr_frame(frame: pd.DataFrame, path: Path) -> None:
    required = {"horizon", "quantile_crossed"}
    assert required <= set(frame), f"missing CQR fields: {path}"
    lower_column, upper_column = (
        ("lower", "upper")
        if {"lower", "upper"} <= set(frame)
        else ("lower_quantile", "upper_quantile")
    )
    assert {lower_column, upper_column} <= set(frame), f"missing CQR bounds: {path}"
    assert frame["quantile_crossed"].notna().all() and frame["quantile_crossed"].eq(False).all(), path
    lower = pd.to_numeric(frame[lower_column], errors="coerce")
    upper = pd.to_numeric(frame[upper_column], errors="coerce")
    assert lower.notna().all() and upper.notna().all() and (lower <= upper).all(), path


def _check_hourly_evidence(path: Path, records: dict[str, dict[str, object]]) -> pd.DataFrame:
    assert path.is_file(), f"missing hourly interval evidence: {path}"
    frame = pd.read_csv(path)
    assert set(frame) == _HOURLY_EVIDENCE_COLUMNS and not frame.empty, path
    frame = frame.loc[:, _HOURLY_EVIDENCE_COLUMN_ORDER]
    assert frame["evaluation_period"].eq("retrospective_final").all(), path
    for column in ("nominal", "empirical_coverage", "mean_width", "interval_score", "n"):
        values = pd.to_numeric(frame[column], errors="coerce")
        assert values.notna().all() and values.map(math.isfinite).all(), f"invalid {column}: {path}"
    n = pd.to_numeric(frame["n"])
    assert (n > 0).all() and (n % 1 == 0).all(), path
    for _, row in frame.iterrows():
        stream_id = str(row["stream_id"])
        assert stream_id in records, f"unregistered stream {stream_id}: {path}"
        assert row["protocol_fingerprint"] == protocol_fingerprint(records[stream_id]), path
        expected_scope = (
            "prequential_monitoring_no_unconditional_time_series_coverage"
            if row["method"] == "adaptive"
            else "empirical_retrospective"
        )
        assert row["coverage_scope"] == expected_scope, path
    grouped = frame[frame["slice_type"].isin({"horizon", "local_day"})]
    assert (pd.to_numeric(grouped["n"]) >= _MIN_HOURLY_GROUP_N).all(), path
    local_days = frame[frame["slice_type"].eq("local_day")]
    assert local_days.empty or pd.to_numeric(local_days["n"]).isin({23, 24, 25}).all(), path
    return frame


def _check_descriptive_study_evidence(results: Path) -> None:
    analysis = results / "analysis"
    weekday_path = analysis / "load_profile_weekday.csv"
    month_path = analysis / "load_profile_month.csv"
    temperature_path = analysis / "temperature_load_curve.csv"
    frames = {
        weekday_path: pd.read_csv(weekday_path),
        month_path: pd.read_csv(month_path),
        temperature_path: pd.read_csv(temperature_path),
    }
    weekday = frames[weekday_path]
    month = frames[month_path]
    temperature = frames[temperature_path]
    assert list(weekday) == _WEEKDAY_COLUMNS and len(weekday) == 7, weekday_path
    assert list(month) == _MONTH_COLUMNS and len(month) == 12, month_path
    assert list(temperature) == _TEMPERATURE_COLUMNS and len(temperature) == 9, temperature_path
    assert weekday["weekday_order"].tolist() == list(range(1, 8)), weekday_path
    assert weekday["weekday"].tolist() == list(calendar.day_name), weekday_path
    assert month["month_order"].tolist() == list(range(1, 13)), month_path
    assert month["month"].tolist() == list(calendar.month_name)[1:], month_path
    assert temperature["bin_order"].tolist() == list(range(1, 10)), temperature_path
    assert temperature["lower_C"].tolist() == list(range(-10, 35, 5)), temperature_path
    assert temperature["upper_C"].tolist() == list(range(-5, 40, 5)), temperature_path
    for path, frame, keys in (
        (weekday_path, weekday, ["weekday_order", "weekday"]),
        (month_path, month, ["month_order", "month"]),
        (temperature_path, temperature, ["bin_order", "lower_C", "upper_C"]),
    ):
        assert not frame.empty and not frame.duplicated(keys).any(), path
        assert frame["evidence_scope"].eq(_DESCRIPTIVE_SCOPE).all(), path
        values = frame[["mean_load_MW", "n_days"]].apply(pd.to_numeric, errors="coerce")
        assert values.notna().all().all() and np.isfinite(values.to_numpy()).all(), path
        assert values["n_days"].gt(0).all() and values["n_days"].mod(1).eq(0).all(), path


def _retrospective_months(record: dict[str, object]) -> list[str]:
    start, end = record["splits"]["retrospective_final"]
    return pd.period_range(start, end, freq="M").astype(str).tolist()


def _check_hourly_monthly_evidence(
    hourly: Path,
    records: dict[str, dict[str, object]],
) -> None:
    coverage_path = hourly / "coverage_by_month.csv"
    residual_path = hourly / "residual_drift_monthly.csv"
    raw_monitor_path = hourly / "residual_drift.csv"
    coverage = _check_hourly_evidence(coverage_path, records)
    assert list(coverage) == _HOURLY_EVIDENCE_COLUMN_ORDER, coverage_path
    assert coverage["slice_type"].eq("month").all(), coverage_path
    methods = ["symmetric", "adaptive", "cqr"]
    assert set(coverage["method"].astype(str)) == set(methods), coverage_path
    month_sets: list[list[str]] = []
    for method in methods:
        rows = coverage[coverage["method"].astype(str).eq(method)]
        assert rows["stream_id"].nunique() == 1, coverage_path
        months = rows["slice_value"].astype(str).tolist()
        assert months == sorted(months) and len(months) == len(set(months)), coverage_path
        assert 1 <= len(months) <= 12, coverage_path
        for stream_id in rows["stream_id"].astype(str).unique():
            assert months == _retrospective_months(records[stream_id]), coverage_path
        month_sets.append(months)
    assert all(months == month_sets[0] for months in month_sets[1:]), coverage_path

    assert residual_path.is_file() and raw_monitor_path.is_file(), residual_path
    residual = pd.read_csv(residual_path)
    assert list(residual) == _MONTHLY_RESIDUAL_COLUMNS and not residual.empty, residual_path
    assert residual["month"].astype(str).tolist() == month_sets[0], residual_path
    assert residual["month"].is_unique, residual_path
    assert residual["evaluation_period"].eq("retrospective_final").all(), residual_path
    assert residual["monitoring_scope"].eq("prequential_monitoring").all(), residual_path
    numeric = residual[
        [
            "mean_absolute_error_MW",
            "max_page_hinkley_statistic",
            "alert_count",
            "n",
        ]
    ].apply(pd.to_numeric, errors="coerce")
    assert numeric.notna().all().all() and np.isfinite(numeric.to_numpy()).all(), residual_path
    assert numeric["n"].gt(0).all() and numeric["n"].mod(1).eq(0).all(), residual_path
    assert numeric["alert_count"].ge(0).all(), residual_path
    assert (numeric["alert_count"] <= numeric["n"]).all(), residual_path
    assert int(numeric["n"].sum()) == len(pd.read_csv(raw_monitor_path)), residual_path
    for row in residual.itertuples():
        stream_id = str(row.stream_id)
        assert stream_id in records, residual_path
        assert row.protocol_fingerprint == protocol_fingerprint(records[stream_id]), residual_path
        assert _retrospective_months(records[stream_id]) == month_sets[0], residual_path


def _check_daily_interval_artifacts(
    results: Path,
    records: dict[str, dict[str, object]],
    point_metadata: dict[str, dict[str, object]],
) -> None:
    """Reject daily interval rows that do not match their registered point stream."""

    interval_dir = results / "interval_predictions"
    summary_path = results / "metrics" / "interval_coverage.csv"
    assert interval_dir.is_dir(), f"missing daily interval predictions: {interval_dir}"
    assert summary_path.is_file(), f"missing daily interval summary: {summary_path}"
    assert {path.name for path in interval_dir.glob("*.csv")} == set(point_metadata), (
        "daily point/interval artifacts differ"
    )

    summary = pd.read_csv(summary_path)
    assert set(summary) == _DAILY_EVIDENCE_COLUMNS and not summary.empty, summary_path
    assert not summary.duplicated(["model", "method", "level"]).any(), summary_path
    for column in (
        "nominal",
        "empirical_coverage",
        "mean_width_MW",
        "interval_score_MW",
        "n",
    ):
        values = pd.to_numeric(summary[column], errors="coerce")
        assert values.notna().all() and values.map(math.isfinite).all(), summary_path
    assert pd.to_numeric(summary["n"]).gt(0).all(), summary_path
    assert set(summary["model"].astype(str)) == {
        Path(name).stem for name in point_metadata
    }, summary_path

    for name, expected_metadata in point_metadata.items():
        path = interval_dir / name
        assert path.is_file(), f"missing daily interval predictions: {path}"
        frame = pd.read_csv(path)
        assert not frame.empty, path
        metadata = _metadata_from_frame(frame, path, "retrospective_final")
        assert metadata == expected_metadata, path
        stream_id = str(metadata["stream_id"])
        assert stream_id in records, f"unregistered stream {stream_id}: {path}"
        assert metadata["protocol_fingerprint"] == protocol_fingerprint(records[stream_id]), path

        model_summary = summary[summary["model"].astype(str).eq(Path(name).stem)]
        assert set(zip(model_summary["method"], model_summary["level"], strict=True)) == {
            ("fixed", "80%"),
            ("fixed", "95%"),
            ("adaptive", "80%"),
            ("adaptive", "95%"),
        }, summary_path
        for _, row in model_summary.iterrows():
            for column in _PROTOCOL_COLUMNS:
                assert row[column] == metadata[column], summary_path
            method = str(row["method"])
            suffix = str(row["level"]).removesuffix("%")
            lower_column = f"fixed_lower_{suffix}" if method == "fixed" else f"lower_{suffix}"
            upper_column = f"fixed_upper_{suffix}" if method == "fixed" else f"upper_{suffix}"
            assert {"actual", lower_column, upper_column} <= set(frame), path
            alpha = 1.0 - float(suffix) / 100.0
            evidence = interval_evidence(
                frame["actual"],
                frame[lower_column],
                frame[upper_column],
                alpha,
            )
            expected_scope = (
                "empirical_retrospective"
                if method == "fixed"
                else "prequential_monitoring_no_unconditional_time_series_coverage"
            )
            assert row["coverage_scope"] == expected_scope, summary_path
            assert math.isclose(float(row["nominal"]), float(evidence["nominal"])), summary_path
            assert math.isclose(
                float(row["empirical_coverage"]),
                round(float(evidence["empirical_coverage"]), 4),
            ), summary_path
            assert math.isclose(
                float(row["mean_width_MW"]),
                round(float(evidence["mean_width"]), 1),
            ), summary_path
            assert math.isclose(
                float(row["interval_score_MW"]),
                round(float(evidence["interval_score"]), 1),
            ), summary_path
            assert int(row["n"]) == evidence["n"], summary_path


def _check_reconciliation_artifacts(hourly: Path, records: dict[str, dict[str, object]]) -> None:
    """Reject stale, non-finite, or incoherent reconciliation evidence."""

    invariants_path = hourly / "reconciliation_invariants.csv"
    metrics_path = hourly / "reconciliation_metrics.csv"
    if not invariants_path.exists() and not metrics_path.exists():
        return
    assert invariants_path.is_file() and metrics_path.is_file(), "incomplete reconciliation evidence"
    invariants = pd.read_csv(invariants_path)
    assert set(invariants) == _RECONCILIATION_INVARIANT_COLUMNS and not invariants.empty, invariants_path
    numeric = ["reconciled_hourly_mean", "daily_anchor", "delta", "abs_delta", "tolerance", "n"]
    for column in numeric:
        values = pd.to_numeric(invariants[column], errors="coerce")
        assert values.notna().all() and values.map(math.isfinite).all(), invariants_path
    assert pd.to_numeric(invariants["n"]).ge(23).all(), invariants_path
    assert pd.to_numeric(invariants["n"]).isin({23, 24, 25}).all(), invariants_path
    assert invariants["pass"].astype(str).str.lower().eq("true").all(), invariants_path
    assert (
        pd.to_numeric(invariants["abs_delta"], errors="coerce")
        <= pd.to_numeric(invariants["tolerance"], errors="coerce")
    ).all(), invariants_path
    assert pd.to_datetime(invariants["local_date"], errors="coerce").notna().all(), invariants_path

    metrics = pd.read_csv(metrics_path)
    assert not metrics.empty and _RECONCILIATION_METADATA <= set(metrics), metrics_path
    assert metrics["evaluation_period"].eq("retrospective_final").all(), metrics_path
    assert metrics[list(_RECONCILIATION_METADATA)].notna().all().all(), metrics_path
    assert metrics[list(_RECONCILIATION_METADATA)].nunique().eq(1).all(), metrics_path
    values = metrics.select_dtypes(include="number")
    assert not values.empty and np.isfinite(values.to_numpy()).all(), metrics_path
    for stream_id, group in metrics.groupby("stream_id"):
        assert str(stream_id) in records, metrics_path
        assert group["protocol_fingerprint"].iat[0] == protocol_fingerprint(records[str(stream_id)]), metrics_path
    assert set(metrics["evaluation_period"]) == {"retrospective_final"}, metrics_path


def _finite_columns(frame: pd.DataFrame, columns: set[str], path: Path) -> None:
    values = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
    assert values.notna().all().all() and np.isfinite(values.to_numpy()).all(), path


def _selection_contract(hourly: Path) -> tuple[str, str, str]:
    path = hourly / "model_selection.csv"
    selection = pd.read_csv(path)
    required = {"selected_model", "selection_metric", "selection_evidence_period"}
    assert len(selection) == 1 and required <= set(selection), path
    row = selection.iloc[0]
    assert row["selection_evidence_period"] == "validation", path
    evidence = "results/hourly/model_ablation_validation_reconciled.csv"
    assert (hourly.parents[1] / evidence).is_file(), path
    return str(row["selection_metric"]), str(row["selected_model"]), evidence


def _check_selection_columns(
    frame: pd.DataFrame,
    path: Path,
    expected: tuple[str, str, str],
) -> None:
    metric, model, evidence = expected
    assert frame["validation_selection_metric"].eq(metric).all(), path
    assert frame["validation_selected_model"].eq(model).all(), path
    assert frame["validation_selection_evidence"].eq(evidence).all(), path


def _hourly_identity_sets(hourly: Path, records: dict[str, dict[str, object]]) -> dict[str, pd.DataFrame]:
    directory = hourly / "test_predictions"
    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(directory.glob("*.csv")):
        frame = pd.read_csv(path)
        required = {*_HOURLY_IDENTITY, "actual", "prediction", "evaluation_period", "stream_id", "protocol_fingerprint"}
        assert not frame.empty and required <= set(frame), path
        for column in ("forecast_origin", "valid_time"):
            frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)
        frame["horizon"] = pd.to_numeric(frame["horizon"], errors="coerce")
        assert frame[[*_HOURLY_IDENTITY, "actual", "prediction"]].notna().all().all(), path
        assert not frame.duplicated(_HOURLY_IDENTITY).any(), path
        assert frame["evaluation_period"].eq("retrospective_final").all(), path
        stream_id = str(frame["stream_id"].iat[0])
        assert stream_id in records and frame["stream_id"].eq(stream_id).all(), path
        assert frame["protocol_fingerprint"].eq(protocol_fingerprint(records[stream_id])).all(), path
        frames[path.stem] = frame
    assert frames, directory
    return frames


def _eligible_hourly_rows(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    common = set.intersection(
        *(set(map(tuple, frame[_HOURLY_IDENTITY].to_numpy())) for frame in frames.values())
    )
    assert common, "hourly comparison artifacts have no common eligible identity"
    return pd.DataFrame(sorted(common), columns=_HOURLY_IDENTITY)


def _comparison_slice(eligible: pd.DataFrame, row: object) -> pd.DataFrame:
    slice_type = str(row.slice_type)
    if slice_type == "aggregate":
        assert str(row.slice_value) == "all"
        return eligible
    if slice_type == "horizon":
        return eligible[eligible["horizon"].eq(int(row.slice_value))]
    local = pd.DatetimeIndex(eligible["valid_time"]).tz_convert("Europe/Berlin")
    if slice_type == "local_hour":
        return eligible[local.hour == int(row.slice_value)]
    if slice_type == "local_day":
        return eligible[pd.Index(local.date).astype(str) == str(row.slice_value)]
    raise AssertionError(f"unsupported hourly comparison slice: {slice_type}")


def _check_comparison_evidence(results: Path, records: dict[str, dict[str, object]]) -> None:
    daily_path = results / "metrics/daily_comparison.csv"
    daily = pd.read_csv(daily_path)
    assert set(daily) == _DAILY_COMPARISON_COLUMNS and not daily.empty, daily_path
    assert daily["evaluation_period"].eq("retrospective_final").all(), daily_path
    _finite_columns(daily, {"n", "MAE", "RMSE", "MAPE", "MASE"}, daily_path)
    for row in daily.itertuples():
        stream_id = str(row.stream_id)
        assert stream_id in records, daily_path
        assert row.protocol_fingerprint == protocol_fingerprint(records[stream_id]), daily_path
        assert int(row.n) == len(pd.read_csv(results / "predictions" / f"{row.model}.csv")), daily_path
        assert row.dates == f"{row.start}:{row.end}", daily_path
        assert pd.Timestamp(row.start) <= pd.Timestamp(row.end), daily_path

    hourly = results / "hourly"
    expected_selection = _selection_contract(hourly)
    raw = _hourly_identity_sets(hourly, records)
    eligible = _eligible_hourly_rows(raw)
    comparison_path = hourly / "hourly_comparison.csv"
    comparison = pd.read_csv(comparison_path)
    assert set(comparison) == _HOURLY_COMPARISON_COLUMNS and not comparison.empty, comparison_path
    assert comparison["evaluation_period"].eq("retrospective_final").all(), comparison_path
    _finite_columns(comparison, {"n", "MAE", "RMSE", "MAPE", "MASE"}, comparison_path)
    _check_selection_columns(comparison, comparison_path, expected_selection)
    for row in comparison.itertuples():
        assert str(row.model) in raw, comparison_path
        stream_id = str(row.stream_id)
        assert stream_id in records, comparison_path
        assert row.protocol_fingerprint == protocol_fingerprint(records[stream_id]), comparison_path
        selected = _comparison_slice(eligible, row)
        assert not selected.empty and int(row.n) == len(selected), comparison_path
        horizons = ",".join(map(str, sorted(pd.to_numeric(selected["horizon"]).astype(int).unique())))
        assert str(row.eligible_horizon) == horizons, comparison_path
        valid_time = pd.DatetimeIndex(selected["valid_time"])
        assert pd.Timestamp(row.local_start) == valid_time.min(), comparison_path
        assert pd.Timestamp(row.local_end) == valid_time.max(), comparison_path

    bootstrap_path = hourly / "model_ablation_test_uncertainty.csv"
    bootstrap = pd.read_csv(bootstrap_path)
    assert set(bootstrap) == _BOOTSTRAP_COLUMNS and not bootstrap.empty, bootstrap_path
    _finite_columns(
        bootstrap,
        {"mae_difference", "ci_lower", "ci_upper", "probability_candidate_better", "n_days", "seed", "n_bootstrap", "block_size_days", "practical_tie_threshold"},
        bootstrap_path,
    )
    _check_selection_columns(bootstrap, bootstrap_path, expected_selection)
    expected_seed = {int(record["seed"]) for record in records.values()}
    assert len(expected_seed) == 1 and set(pd.to_numeric(bootstrap["seed"]).astype(int)) == expected_seed, bootstrap_path
    assert pd.to_numeric(bootstrap["n_bootstrap"]).gt(0).all(), bootstrap_path
    assert pd.to_numeric(bootstrap["block_size_days"]).gt(0).all(), bootstrap_path
    assert bootstrap["uncertainty_scope"].str.contains("retrospective_final", regex=False).all(), bootstrap_path
    assert bootstrap["selected_model"].eq(expected_selection[1]).all(), bootstrap_path
    assert set(bootstrap["candidate"]).union(bootstrap["reference"]) <= set(raw), bootstrap_path
    local_days = pd.DatetimeIndex(eligible["valid_time"]).tz_convert("Europe/Berlin").date
    assert pd.to_numeric(bootstrap["n_days"]).eq(len(set(local_days))).all(), bootstrap_path


def validate_protocol_contract(
    results: Path,
    ledger_path: Path,
    *,
    calibration_start=None,
    calibration_end=None,
    final_start=None,
    final_end=None,
    source_root: Path | None = None,
    config_path: Path | None = None,
) -> None:
    """Fail closed unless the persisted protocol and interval evidence agree."""

    _require_assertions_enabled()
    _check_ledger(ledger_path)
    records = _load_protocol_manifest(results)
    _check_descriptive_study_evidence(results)
    calibration = results / "calibration_predictions"
    final = results / "predictions"
    assert calibration.is_dir() and final.is_dir(), "missing daily calibration/final artifacts"
    assert {path.name for path in calibration.glob("*.csv")} == {
        path.name for path in final.glob("*.csv")
    }, "daily calibration/final artifacts differ"
    daily_metadata = {}
    for path in sorted(calibration.glob("*.csv")):
        _, _, daily_metadata[path.name] = _check_registered_pair(
            path,
            final / path.name,
            records,
        )
    _check_daily_interval_artifacts(results, records, daily_metadata)

    hourly = results / "hourly"
    _check_hourly_monthly_evidence(hourly, records)
    _, point_final, point_metadata = _check_registered_pair(
        hourly / "calibration_predictions.csv", hourly / "test_intervals.csv", records
    )
    cqr_calibration, cqr_final, _ = _check_registered_pair(
        hourly / "cqr_calibration_predictions.csv", hourly / "cqr_test_intervals.csv", records
    )
    _check_cqr_frame(cqr_calibration, hourly / "cqr_calibration_predictions.csv")
    _check_cqr_frame(cqr_final, hourly / "cqr_test_intervals.csv")
    _check_reconciliation_artifacts(hourly, records)
    _check_comparison_evidence(results, records)
    if any(value is not None for value in (calibration_start, calibration_end, final_start, final_end)):
        _check_hourly_predictions(
            hourly / "calibration_predictions.csv",
            required_columns=("actual", "prediction"),
            expected_start=calibration_start,
            expected_end=calibration_end,
        )
        _check_hourly_predictions(
            hourly / "test_intervals.csv",
            required_columns=("actual", "prediction", "lower", "upper"),
            expected_start=final_start,
            expected_end=final_end,
        )
        _check_hourly_predictions(
            hourly / "cqr_calibration_predictions.csv",
            required_columns=("actual", "prediction", "lower_quantile", "upper_quantile"),
            expected_start=calibration_start,
            expected_end=calibration_end,
        )
        _check_hourly_predictions(
            hourly / "cqr_test_intervals.csv",
            required_columns=("actual", "prediction", "lower", "upper"),
            expected_start=final_start,
            expected_end=final_end,
        )
    calibration_horizons = set(pd.to_numeric(cqr_calibration["horizon"]).astype(int))
    final_horizons = set(pd.to_numeric(cqr_final["horizon"]).astype(int))
    assert final_horizons <= calibration_horizons, "CQR final horizon lacks calibration scores"

    aggregate = _check_hourly_evidence(hourly / "interval_comparison.csv", records)
    horizon = _check_hourly_evidence(hourly / "coverage_by_horizon.csv", records)
    local_day = _check_hourly_evidence(hourly / "coverage_by_local_day.csv", records)
    final_intervals = {"symmetric": point_final, "adaptive": point_final, "cqr": cqr_final}
    for frame, expected_slice in ((aggregate, "aggregate"), (horizon, "horizon"), (local_day, "local_day")):
        assert frame["slice_type"].eq(expected_slice).all(), f"wrong slice in {expected_slice} evidence"
        assert set(frame["method"]) == {"symmetric", "adaptive", "cqr"}, frame
        for row in frame.itertuples():
            selected = _comparison_slice(final_intervals[str(row.method)], row)
            assert not selected.empty and int(row.n) == len(selected), frame
    emitted_horizons = set(pd.to_numeric(cqr_final["horizon"], errors="coerce").dropna().astype(int))
    evidence_horizons = set(pd.to_numeric(horizon["slice_value"], errors="coerce").dropna().astype(int))
    assert evidence_horizons == emitted_horizons, horizon
    assert point_metadata["evaluation_period"] == "retrospective_final"

    summary_path = results / "run_summary.json"
    assert summary_path.is_file(), f"missing run summary: {summary_path}"
    summary = json.loads(summary_path.read_text())
    assert summary["final_role"] == "retrospective_final", summary_path
    assert summary["protocol_manifest"] == "evaluation_protocol.json", summary_path
    assert summary["protocol_schema_version"] == 1, summary_path
    expected_fingerprints = {stream_id: protocol_fingerprint(record) for stream_id, record in records.items()}
    assert summary["protocol_fingerprints"] == expected_fingerprints, summary_path
    assert summary["retrospective_final_metrics"] == summary["test_metrics"], summary_path
    if (source_root is None) != (config_path is None):
        raise ValueError("source_root and config_path must be provided together")
    if source_root is not None and config_path is not None:
        _validate_release_manifest(summary, results, records, source_root, config_path)


def _check_managed_presentations(root: Path, results: Path) -> None:
    """Require exact generated blocks and qualified authored claims on every surface."""

    values = load_presentation_values(results)
    for relative in MANAGED_PRESENTATION_PATHS:
        path = root / relative
        if not path.is_file():
            raise ValueError(f"missing managed presentation surface: {path}")
        check_managed_presentation(path.read_text(encoding="utf-8"), values, relative)


def main() -> None:
    _require_assertions_enabled()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = Config.from_yaml(config_path)
    results = cfg.path("results_dir")
    source_root_result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=config_path.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    source_root = Path(source_root_result.stdout.strip())
    validate_protocol_contract(
        results,
        cfg.root / "docs" / "EVALUATION_LEDGER.md",
        calibration_start=cfg.split.calibration_start,
        calibration_end=cfg.split.calibration_end,
        final_start=cfg.split.test_start,
        final_end=cfg.split.test_end,
        source_root=source_root,
        config_path=config_path,
    )

    dataset = pd.read_parquet(cfg.path("processed_dir") / "dataset.parquet")
    features = pd.read_parquet(cfg.path("processed_dir") / "features.parquet")
    assert dataset.index.min() == cfg.dataset_start
    assert dataset.index.max() == cfg.raw_end
    assert {"Temp_forecast", "Wind_forecast"} <= set(features)
    assert not {"Temp_t", "Wind_t"} & set(features)
    assert features.iloc[1]["Temp_forecast"] == dataset.iloc[0]["Temp_t"]
    operational_start = pd.Timestamp(cfg.weather["operational_start"]).date()
    assert (
        features.loc[operational_start, "Temp_forecast"]
        == dataset.loc[operational_start, "Temp_operational"]
    )

    validation_metrics = pd.read_csv(results / "metrics" / "validation_metrics.csv", index_col=0)
    calibration_metrics = pd.read_csv(results / "metrics" / "calibration_metrics.csv", index_col=0)
    test_metrics = pd.read_csv(results / "metrics" / "test_metrics.csv", index_col=0)
    assert set(validation_metrics.index) == MODELS
    assert set(calibration_metrics.index) == MODELS
    assert set(test_metrics.index) == MODELS

    validation_start = cfg.split.train_end + timedelta(days=1)
    validation_rows = (cfg.split.val_end - validation_start).days + 1
    calibration_rows = (cfg.split.calibration_end - cfg.split.calibration_start).days + 1
    test_rows = (cfg.split.test_end - cfg.split.test_start).days + 1
    for model in MODELS:
        _check_predictions(
            results / "validation_predictions" / f"{model}.csv",
            validation_start,
            cfg.split.val_end,
            validation_rows,
        )
        _check_predictions(
            results / "calibration_predictions" / f"{model}.csv",
            cfg.split.calibration_start,
            cfg.split.calibration_end,
            calibration_rows,
        )
        _check_predictions(
            results / "predictions" / f"{model}.csv",
            cfg.split.test_start,
            cfg.split.test_end,
            test_rows,
        )
        interval_prediction = _check_predictions(
            results / "interval_predictions" / f"{model}.csv",
            cfg.split.test_start,
            cfg.split.test_end,
            test_rows,
            required_columns=(
                "actual",
                "forecast",
                "fixed_lower_80",
                "fixed_upper_80",
                "lower_80",
                "upper_80",
                "adaptive_alpha_80",
                "fixed_lower_95",
                "fixed_upper_95",
                "lower_95",
                "upper_95",
                "adaptive_alpha_95",
            ),
        )
        assert (interval_prediction["lower_80"] <= interval_prediction["forecast"]).all()
        assert (interval_prediction["forecast"] <= interval_prediction["upper_80"]).all()
        assert (interval_prediction["lower_95"] <= interval_prediction["lower_80"]).all()
        assert (interval_prediction["upper_80"] <= interval_prediction["upper_95"]).all()

    hourly = results / "hourly"
    _check_hourly_candidates(
        hourly,
        metrics_name="model_ablation_validation.csv",
        artifacts_name="validation_predictions",
    )
    _check_hourly_candidates(
        hourly,
        metrics_name="model_ablation_validation_reconciled.csv",
        artifacts_name="validation_reconciled_predictions",
    )
    _check_hourly_candidates(
        hourly,
        metrics_name="model_ablation_test.csv",
        artifacts_name="test_predictions",
    )
    _check_hourly_candidates(
        hourly,
        metrics_name="model_ablation_test_reconciled.csv",
        artifacts_name="test_reconciled_predictions",
    )
    _check_hourly_predictions(
        hourly / "calibration_predictions.csv", required_columns=("actual", "prediction")
    )
    _check_hourly_predictions(
        hourly / "test_intervals.csv", required_columns=("actual", "prediction", "lower", "upper")
    )
    _check_hourly_predictions(
        hourly / "cqr_calibration_predictions.csv",
        required_columns=("actual", "prediction", "lower_quantile", "upper_quantile"),
    )
    _check_hourly_predictions(
        hourly / "cqr_test_intervals.csv", required_columns=("actual", "prediction", "lower", "upper")
    )
    _check_hourly_predictions(
        hourly / "test_reconciled_predictions.csv", required_columns=("actual", "prediction")
    )

    summary = json.loads((results / "run_summary.json").read_text())
    assert summary["project_version"] == __version__
    assert summary["protocol"]["weather_strategy"] == "available_day_ahead"
    assert summary["protocol"]["calibration_end"] == cfg.split.calibration_end.isoformat()
    assert summary["protocol"]["test_end"] == cfg.split.test_end.isoformat()
    assert summary["protocol"]["final_role"] == "retrospective_final"
    dataset_file = cfg.path("processed_dir") / "dataset.parquet"
    assert summary["data"]["rows"] == len(dataset)
    assert summary["data"]["dataset_sha256"] == hashlib.sha256(dataset_file.read_bytes()).hexdigest()
    for key, expected in [
        ("validation_metrics", validation_metrics),
        ("calibration_metrics", calibration_metrics),
        ("retrospective_final_metrics", test_metrics),
        ("test_metrics", test_metrics),
    ]:
        recorded = pd.DataFrame.from_dict(summary[key], orient="index")
        pd.testing.assert_frame_equal(
            recorded.sort_index().sort_index(axis=1),
            expected.sort_index().sort_index(axis=1),
            check_names=False,
        )

    _check_managed_presentations(cfg.root, results)

    markdown_files = [
        cfg.root / "README.md",
        cfg.root / "docs" / "TECHNICAL.md",
        cfg.root / "report" / "technical-report-en.md",
    ]
    markdown_text = "\n".join(path.read_text(encoding="utf-8") for path in markdown_files)
    assert r"\(" not in markdown_text
    assert r"\)" not in markdown_text
    assert "$$" not in markdown_text
    assert markdown_text.count("```math") == 2

    print(
        f"results valid: {len(dataset)} data rows, "
        f"{validation_rows} validation days, {calibration_rows} calibration days, "
        f"{test_rows} test days"
    )


if __name__ == "__main__":
    main()
