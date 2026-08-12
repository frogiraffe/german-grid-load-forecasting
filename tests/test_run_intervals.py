from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from loadfc.evaluation.protocol import protocol_fingerprint

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_intervals.py"
CONFIG = Path(__file__).parent / "fixtures" / "config_test.yaml"


def _module():
    spec = importlib.util.spec_from_file_location("run_intervals", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prediction_csv_uses_python_parser(tmp_path, monkeypatch):
    module = _module()
    expected = pd.DataFrame({"actual": [1.0], "forecast": [1.0]})
    calls: list[tuple[Path, dict[str, object]]] = []

    def read_csv(path: Path, **kwargs):
        calls.append((path, kwargs))
        return expected

    monkeypatch.setattr(module.pd, "read_csv", read_csv)
    path = tmp_path / "predictions.csv"

    assert module._read_prediction_csv(path) is expected
    assert calls == [
        (path, {"index_col": 0, "parse_dates": [0], "engine": "python"}),
    ]


def _record() -> dict[str, object]:
    return {
        "schema_version": 1,
        "source_revision": "abc123",
        "config_sha256": "config-sha",
        "stream_id": "daily/SARIMAX",
        "model_identity": "SARIMAX",
        "ordered_feature_columns": ["L_t-1", "Temp_forecast"],
        "model_parameters": {"order": [2, 1, 1], "refit": "false"},
        "seed": 42,
        "splits": {
            "train": ["2019-01-14", "2023-12-31"],
            "validation": ["2024-01-01", "2025-06-30"],
            "calibration": ["2025-07-01", "2025-12-31"],
            "retrospective_final": ["2026-01-01", "2026-08-04"],
        },
        "weather_strategy": "available_day_ahead",
        "point_state_policy": {"fit_through": "2025-06-30", "update": "after_actual"},
        "interval_state_policy": {"adaptive": "updates_after_actual_only"},
        "final_role": "retrospective_final",
        "validation_selection_evidence": "results/metrics/validation_metrics.csv",
        "rationale": "Validation metrics select the configured candidate; final is retrospective.",
    }


def _artifact(record: dict[str, object], period: str) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=3, freq="D")
    return pd.DataFrame(
        {
            # Canonical daily values come from averaging hourly rows, so they carry a
            # non-terminating expansion. Shifting actual and forecast by the same
            # amount leaves every residual, width, and coverage expectation unchanged
            # while making any rounding of the written artifact observable.
            "actual": [10 + 1 / 3, 13 + 1 / 3, 8 + 1 / 3],
            "forecast": [9 + 1 / 3, 11 + 1 / 3, 9 + 1 / 3],
            "forecast_origin": dates - pd.Timedelta(days=1),
            "valid_time": dates,
            "weather_source_run": ["persistence"] * 3,
            "weather_availability_assumption": ["previous-day realized"] * 3,
            "stream_id": record["stream_id"],
            "protocol_fingerprint": protocol_fingerprint(record),
            "evaluation_period": period,
            "point_state_policy": json.dumps(record["point_state_policy"], sort_keys=True),
            "interval_state_policy": json.dumps(record["interval_state_policy"], sort_keys=True),
        },
        index=dates,
    )


def _write_stream(results_dir: Path, *, changed: tuple[str, object] | None = None) -> None:
    record = _record()
    results_dir.mkdir(parents=True)
    (results_dir / "evaluation_protocol.json").write_text(
        json.dumps({"schema_version": 1, "records": {record["stream_id"]: record}})
    )
    calibration = _artifact(record, "calibration")
    final = _artifact(record, "retrospective_final")
    if changed is not None:
        field, value = changed
        final[field] = value
    calibration_dir = results_dir / "calibration_predictions"
    predictions_dir = results_dir / "predictions"
    calibration_dir.mkdir(parents=True)
    predictions_dir.mkdir()
    calibration.to_csv(calibration_dir / "SARIMAX.csv")
    final.to_csv(predictions_dir / "SARIMAX.csv")


def _run(tmp_path: Path, monkeypatch) -> Path:
    results_dir = tmp_path / "results"
    module = _module()
    module.MODELS = ["SARIMAX"]
    monkeypatch.setattr(module, "artifact_results_root", lambda cfg: results_dir)
    monkeypatch.setattr(sys, "argv", ["run_intervals.py", "--config", str(CONFIG)])
    module.main()
    return results_dir


def test_run_intervals_writes_complete_fixed_evidence_for_a_registered_stream(tmp_path, monkeypatch):
    _write_stream(tmp_path / "results")

    results_dir = _run(tmp_path, monkeypatch)

    evidence = pd.read_csv(results_dir / "metrics" / "interval_coverage.csv")
    fixed = evidence[(evidence["method"] == "fixed") & (evidence["level"] == "80%")].iloc[0]
    assert {
        "model",
        "method",
        "level",
        "nominal",
        "empirical_coverage",
        "mean_width_MW",
        "interval_score_MW",
        "n",
        "evaluation_period",
        "stream_id",
        "protocol_fingerprint",
        "coverage_scope",
    } <= set(evidence)
    assert fixed["evaluation_period"] == "retrospective_final"
    assert fixed["coverage_scope"] == "empirical_retrospective"
    assert fixed["n"] == 3

    intervals = pd.read_csv(results_dir / "interval_predictions" / "SARIMAX.csv")
    for column in (
        "stream_id",
        "protocol_fingerprint",
        "evaluation_period",
        "point_state_policy",
        "interval_state_policy",
    ):
        assert intervals[column].nunique() == 1
        assert intervals[column].iat[0] == fixed[column]


def test_run_intervals_adaptive_evidence_is_prequential_and_preserves_point_artifact(
    tmp_path, monkeypatch
):
    results_dir = tmp_path / "results"
    _write_stream(results_dir)
    source = pd.read_csv(results_dir / "predictions" / "SARIMAX.csv")

    results_dir = _run(tmp_path, monkeypatch)

    evidence = pd.read_csv(results_dir / "metrics" / "interval_coverage.csv")
    adaptive = evidence[(evidence["method"] == "adaptive") & (evidence["level"] == "80%")].iloc[0]
    assert adaptive["coverage_scope"] == (
        "prequential_monitoring_no_unconditional_time_series_coverage"
    )
    assert {
        "nominal",
        "empirical_coverage",
        "mean_width_MW",
        "interval_score_MW",
        "n",
    } <= set(adaptive.index)
    assert adaptive["n"] == 3

    intervals = pd.read_csv(results_dir / "interval_predictions" / "SARIMAX.csv")
    point_columns = [
        "actual",
        "forecast",
        "forecast_origin",
        "valid_time",
        "weather_source_run",
        "weather_availability_assumption",
        "stream_id",
        "protocol_fingerprint",
        "evaluation_period",
        "point_state_policy",
        "interval_state_policy",
    ]
    # Exact: downstream evidence compares these columns with DataFrame.equals, so a
    # rounded interval artifact silently breaks the release even though it is close.
    pd.testing.assert_frame_equal(
        intervals[point_columns],
        source[point_columns],
        check_dtype=False,
        check_exact=True,
    )


def test_run_intervals_adaptive_miss_changes_only_later_bounds(tmp_path, monkeypatch):
    baseline_root = tmp_path / "baseline"
    changed_root = tmp_path / "changed"
    _write_stream(baseline_root / "results")
    _write_stream(changed_root / "results")
    changed_predictions = changed_root / "results" / "predictions" / "SARIMAX.csv"
    changed = pd.read_csv(changed_predictions, index_col=0)
    changed.loc[changed.index[1], "actual"] = 1_000.0
    changed.to_csv(changed_predictions)

    baseline_results = _run(baseline_root, monkeypatch)
    changed_results = _run(changed_root, monkeypatch)
    baseline = pd.read_csv(baseline_results / "interval_predictions" / "SARIMAX.csv")
    changed = pd.read_csv(changed_results / "interval_predictions" / "SARIMAX.csv")

    pd.testing.assert_series_equal(baseline["forecast"], changed["forecast"])
    pd.testing.assert_series_equal(
        baseline.loc[:1, "lower_80"], changed.loc[:1, "lower_80"], check_names=False
    )
    pd.testing.assert_series_equal(
        baseline.loc[:1, "upper_80"], changed.loc[:1, "upper_80"], check_names=False
    )
    assert baseline.loc[2, "lower_80"] != changed.loc[2, "lower_80"]
    assert baseline.loc[2, "upper_80"] != changed.loc[2, "upper_80"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("protocol_fingerprint", "different", "protocol_fingerprint"),
        ("point_state_policy", '{"fit_through":"other"}', "point_state_policy"),
        ("interval_state_policy", '{"fixed":"other"}', "interval_state_policy"),
        ("evaluation_period", "calibration", "evaluation_period"),
    ],
)
def test_run_intervals_rejects_incompatible_stream_before_writing_evidence(
    tmp_path, monkeypatch, field, value, message
):
    _write_stream(tmp_path / "results", changed=(field, value))

    with pytest.raises(ValueError, match=message):
        _run(tmp_path, monkeypatch)

    assert not (tmp_path / "results" / "metrics" / "interval_coverage.csv").exists()
