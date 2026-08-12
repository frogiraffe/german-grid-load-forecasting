from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from loadfc import presentation
from loadfc.evaluation.protocol import protocol_fingerprint
from loadfc.presentation import MANAGED_PRESENTATION_PATHS
from loadfc.tracking import bundle_fingerprint
from scripts import render_report_data


class _Config:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, key: str) -> Path:
        assert key == "results_dir"
        return self.root / "results"


def _record(stream_id: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "source_revision": "fixture-revision",
        "config_sha256": "fixture-config",
        "stream_id": stream_id,
        "model_identity": stream_id,
        "ordered_feature_columns": ["feature"],
        "model_parameters": {"seed": 42},
        "seed": 42,
        "splits": {
            "train": ["2019-01-14", "2023-12-31"],
            "validation": ["2024-01-01", "2025-06-30"],
            "calibration": ["2025-07-01", "2025-12-31"],
            "retrospective_final": ["2026-01-01", "2026-08-04"],
        },
        "weather_strategy": "available_day_ahead",
        "point_state_policy": {"fit_through": "2025-06-30"},
        "interval_state_policy": {"fixed": "frozen"},
        "final_role": "retrospective_final",
        "validation_selection_evidence": "results/metrics/validation_metrics.csv",
        "rationale": "validation-only fixture selection",
    }


def _write_report_inputs(root: Path) -> tuple[Path, Path]:
    metrics = root / "results" / "metrics"
    hourly = root / "results" / "hourly"
    report = root / "report"
    metrics.mkdir(parents=True)
    hourly.mkdir(parents=True)
    report.mkdir()

    model_metrics = pd.DataFrame(
        {
            "RMSE": [2.0, 1.5, 4.0, 3.0],
            "MAE": [1.5, 1.0, 3.0, 2.5],
            "MAPE": [2.5, 2.0, 7.0, 5.5],
            "MASE": [0.6, 0.5, 1.6, 1.3],
        },
        index=["SARIMAX", "ensemble", "naive_1d", "seasonal_naive_7d"],
    )
    for name in ("validation_metrics.csv", "calibration_metrics.csv", "test_metrics.csv"):
        model_metrics.to_csv(metrics / name)
    pd.DataFrame(
        {
            "SARIMAX": [2.5],
            "xgboost": [2.4],
            "lightgbm": [2.3],
            "random_forest": [2.2],
            "seasonal_naive_7d": [3.0],
        },
        index=["2025-H1"],
    ).to_csv(metrics / "rolling_origin_mape.csv")
    pd.DataFrame(
        {
            "operational_MAPE": [2.0],
            "persistence_MAPE": [2.1],
            "delta_pp": [-0.1],
        },
        index=["ensemble"],
    ).to_csv(metrics / "weather_ablation.csv")
    pd.DataFrame(
        {"validation_MAPE": [2.1], "test_MAPE": [2.0], "absolute_gap_pp": [-0.1]},
        index=["ensemble"],
    ).to_csv(metrics / "generalization_gap.csv")

    records = {
        record["stream_id"]: record
        for record in (
            _record("daily/ensemble"),
            _record("daily/lightgbm"),
            _record("daily/naive_1d"),
            _record("daily/seasonal_naive_7d"),
            _record("hourly/point/residual_hybrid"),
            _record("hourly/quantile/lightgbm_cqr"),
        )
    }
    fingerprints = {stream_id: protocol_fingerprint(record) for stream_id, record in records.items()}
    (root / "results/evaluation_protocol.json").write_text(
        json.dumps({"schema_version": 1, "records": records})
    )
    pd.DataFrame(
        [
            {
                "model": model,
                "evaluation_period": "retrospective_final",
                "stream_id": f"daily/{model}",
                "protocol_fingerprint": fingerprints[f"daily/{model}"],
                "dates": "2026-01-01:2026-08-04",
                "start": "2026-01-01",
                "end": "2026-08-04",
                "n": 216,
                **model_metrics.loc[model].to_dict(),
            }
            for model in ("ensemble", "naive_1d", "seasonal_naive_7d")
        ]
    ).to_csv(metrics / "daily_comparison.csv", index=False)
    (metrics / "low_risk_improvement_decision.json").write_text(
        json.dumps(
            {
                "selected_model": "ensemble",
                "decision": "accepted",
                "protocol_fingerprint": fingerprints["daily/ensemble"],
                "integrity_checks": {"no_final_outcomes_used": True},
            }
        )
    )

    daily = pd.DataFrame(
        [
            {
                "model": "ensemble",
                "method": method,
                "level": "95%",
                "evaluation_period": "retrospective_final",
                "stream_id": "daily-point-v1",
                "protocol_fingerprint": "daily-fingerprint",
                "point_state_policy": "{}",
                "interval_state_policy": "{}",
                "coverage_scope": scope,
                "nominal": 0.95,
                "empirical_coverage": coverage,
                "mean_width_MW": width,
                "interval_score_MW": score,
                "n": 216,
            }
            for method, scope, coverage, width, score in (
                ("fixed", "empirical_retrospective", 0.91, 4517.7, 4700.5),
                (
                    "adaptive",
                    "prequential_monitoring_no_unconditional_time_series_coverage",
                    0.95,
                    5054.4,
                    5200.5,
                ),
            )
        ]
    )
    daily.to_csv(metrics / "interval_coverage.csv", index=False)

    pd.DataFrame(
        [
            {
                "selected_model": "residual_hybrid",
                "daily_anchor_model": "ensemble",
                "selection_evidence_period": "validation",
                "selection_protocol_fingerprint": fingerprints[
                    "hourly/point/residual_hybrid"
                ],
            }
        ]
    ).to_csv(hourly / "model_selection.csv", index=False)
    hourly_metrics = pd.DataFrame(
        {"hourly_MAE": [1444.8, 1453.7]},
        index=["residual_hybrid", "lightgbm_direct"],
    )
    hourly_metrics.index.name = "model"
    hourly_metrics.to_csv(hourly / "model_ablation_validation_reconciled.csv")
    hourly_metrics.to_csv(hourly / "model_ablation_test_reconciled.csv")
    pd.DataFrame(
        {"hourly_MAE": [1600.0, 1590.0], "n_hours": [5184, 5184]},
        index=["residual_hybrid", "lightgbm_direct"],
    ).rename_axis("model").to_csv(hourly / "model_ablation_test.csv")
    pd.DataFrame(
        {"hourly_MAE": [1500.0, 1490.0], "n_hours": [5184, 5184]},
        index=["residual_hybrid", "lightgbm_direct"],
    ).rename_axis("model").to_csv(hourly / "model_ablation_test_reconciled.csv")
    pd.DataFrame(
        [
            {
                "candidate": "lightgbm_direct",
                "reference": "residual_hybrid",
                "mae_difference": -10.0,
                "ci_lower": -20.0,
                "ci_upper": 5.0,
                "probability_candidate_better": 0.8,
                "n_days": 216,
                "selected_model": "residual_hybrid",
                "selected_model_preserved": True,
                "validation_selected_model": "residual_hybrid",
            },
            {
                "candidate": "ridge_direct",
                "reference": "residual_hybrid",
                "mae_difference": 100.0,
                "ci_lower": 50.0,
                "ci_upper": 150.0,
                "probability_candidate_better": 0.0,
                "n_days": 216,
                "selected_model": "residual_hybrid",
                "selected_model_preserved": False,
                "validation_selected_model": "residual_hybrid",
            },
        ]
    ).to_csv(hourly / "model_ablation_test_uncertainty.csv", index=False)
    reconciliation = pd.DataFrame(
        {
            "MAE": [1600.0, 1500.0],
            "n_hours": [5184, 5184],
            "evaluation_period": ["retrospective_final", "retrospective_final"],
            "stream_id": ["hourly/point/residual_hybrid"] * 2,
            "protocol_fingerprint": [fingerprints["hourly/point/residual_hybrid"]] * 2,
        },
        index=["residual_hybrid_raw", "residual_hybrid_reconciled"],
    )
    reconciliation.index.name = "model"
    reconciliation.to_csv(hourly / "reconciliation_metrics.csv")

    hourly_intervals = pd.DataFrame(
        [
            {
                "method": method,
                "level": "90%",
                "slice_type": "aggregate",
                "slice_value": "all",
                "evaluation_period": "retrospective_final",
                "coverage_scope": scope,
                "stream_id": (
                    "hourly/quantile/lightgbm_cqr"
                    if method == "cqr"
                    else "hourly/point/residual_hybrid"
                ),
                "protocol_fingerprint": fingerprints[
                    "hourly/quantile/lightgbm_cqr"
                    if method == "cqr"
                    else "hourly/point/residual_hybrid"
                ],
                "nominal": 0.9,
                "empirical_coverage": coverage,
                "mean_width": width,
                "interval_score": score,
                "n": 5184,
            }
            for method, scope, coverage, width, score in (
                ("symmetric", "empirical_retrospective", 0.94, 6700.0, 7000.0),
                (
                    "adaptive",
                    "prequential_monitoring_no_unconditional_time_series_coverage",
                    0.92,
                    6600.0,
                    7100.0,
                ),
                ("cqr", "empirical_retrospective", 0.89, 6800.0, 7200.0),
            )
        ]
    )
    hourly_intervals.to_csv(hourly / "interval_comparison.csv", index=False)

    identity = {
        "source_revision": "fixture-revision",
        "source_clean": True,
        "config_sha256": "fixture-config",
        "lock_sha256": "a" * 64,
        "protocol_fingerprints": fingerprints,
        "declared_artifacts": {
            "dashboard/src/generated/release.json": None,
            "report/generated_results.tex": None,
            "report/technical-report-en.pdf": None,
            "results/metrics/daily_comparison.csv": "b" * 64,
        },
    }
    (root / "results/run_summary.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "final_role": "retrospective_final",
                **identity,
                "bundle_fingerprint": bundle_fingerprint(identity),
            }
        )
    )
    for relative in MANAGED_PRESENTATION_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".tex":
            markers = "% loadfc:generated-start\nstale\n% loadfc:generated-end"
        else:
            markers = "<!-- loadfc:generated-start -->\nstale\n<!-- loadfc:generated-end -->"
        path.write_text(f"authored before\n{markers}\nauthored after\n")
    return metrics / "interval_coverage.csv", hourly / "interval_comparison.csv"


def _run(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(render_report_data.Config, "from_yaml", lambda _: _Config(root))
    monkeypatch.setattr(sys, "argv", ["render_report_data.py"])
    render_report_data.main()
    return root / "report" / "generated_results.tex"


def _check(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(render_report_data.Config, "from_yaml", lambda _: _Config(root))
    monkeypatch.setattr(sys, "argv", ["render_report_data.py", "--check"])
    render_report_data.main()


def test_renders_canonical_daily_and_hourly_interval_evidence(tmp_path, monkeypatch):
    _write_report_inputs(tmp_path)

    destination = _run(tmp_path, monkeypatch)

    text = destination.read_text()
    assert "Ensemble & Fixed & 95\\% & 0.95 & 0.910 & 4517.7 & 4700.5 & 216" in text
    assert r"\newcommand{\SymmetricNominalLevel}{90\%}" in text
    assert r"\newcommand{\SymmetricCoverage}{94.00\%}" in text
    assert r"\newcommand{\SymmetricWidth}{6700.0}" in text
    assert r"\newcommand{\SymmetricIntervalScore}{7000.0}" in text
    assert r"\newcommand{\SymmetricN}{5184}" in text
    assert r"\newcommand{\CQRNominalLevel}{90\%}" in text
    assert r"\newcommand{\CQRCoverage}{89.00\%}" in text
    assert "Empirical retrospective coverage." in text
    assert "Prequential monitoring evidence; no unconditional time-series coverage guarantee." in text


def test_one_release_fixture_renders_the_same_results_across_report_and_readme(
    tmp_path, monkeypatch
):
    _write_report_inputs(tmp_path)

    destination = _run(tmp_path, monkeypatch)

    report = destination.read_text()
    readme = (tmp_path / "README.md").read_text()
    fingerprint = json.loads((tmp_path / "results/run_summary.json").read_text())[
        "bundle_fingerprint"
    ]
    for expected in (
        "2.000",
        "1500.0",
        "89.00",
        "5184",
        "fixture-revision",
    ):
        assert expected in report
        assert expected in readme
    assert r"\newcommand{\FinalRole}{retrospective\_final}" in report
    assert r"\newcommand{\FinalStart}{2026-01-01}" in report
    assert r"\newcommand{\FinalEnd}{2026-08-04}" in report
    assert rf"\newcommand{{\BundleFingerprint}}{{{fingerprint}}}" in report
    assert "**Period:** 01 Jan 2026 to 04 Aug 2026" in readme
    assert "The final period was already examined" in readme
    assert "`retrospective_final`" not in readme
    assert "01 Jan 2026 to 04 Aug 2026" in readme
    assert "**Hourly forecast:** Residual hybrid" in readme
    assert "prequential monitoring" not in readme
    for relative in MANAGED_PRESENTATION_PATHS:
        surface = (tmp_path / relative).read_text()
        assert f"bundle `{fingerprint}`" in surface or f"bundle {fingerprint}" in surface
        assert surface.startswith("authored before\n")
        assert surface.endswith("\nauthored after\n")
        assert surface.count("loadfc:generated-start") == 1
        assert surface.count("loadfc:generated-end") == 1


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("missing", None),
        ("short", "a" * 63),
        ("uppercase", "A" * 64),
        ("non_hex", "g" * 64),
        ("identity_conflict", "0" * 64),
    ],
)
def test_invalid_bundle_fingerprint_fails_before_any_presentation_write(
    tmp_path, monkeypatch, mutation, value
):
    _write_report_inputs(tmp_path)
    summary_path = tmp_path / "results/run_summary.json"
    summary = json.loads(summary_path.read_text())
    if mutation == "missing":
        summary.pop("bundle_fingerprint")
    else:
        summary["bundle_fingerprint"] = value
    summary_path.write_text(json.dumps(summary))
    managed = [tmp_path / relative for relative in MANAGED_PRESENTATION_PATHS]
    before = {path: path.read_bytes() for path in managed}

    with pytest.raises(ValueError, match="bundle fingerprint"):
        _run(tmp_path, monkeypatch)

    assert not (tmp_path / "report/generated_results.tex").exists()
    assert {path: path.read_bytes() for path in managed} == before


def test_check_mode_reports_drift_without_writing(tmp_path, monkeypatch):
    _write_report_inputs(tmp_path)
    _run(tmp_path, monkeypatch)
    readme = tmp_path / "README.md"
    readme.write_text(readme.read_text().replace("MAPE 2.000%", "MAPE 9.999%"))
    paths = [tmp_path / relative for relative in MANAGED_PRESENTATION_PATHS]
    paths.append(tmp_path / "report/generated_results.tex")
    before = {path: path.read_bytes() for path in paths}

    with pytest.raises(ValueError, match="presentation drift"):
        _check(tmp_path, monkeypatch)

    assert {path: path.read_bytes() for path in paths} == before


@pytest.mark.parametrize(
    ("artifact", "column", "value"),
    [
        ("daily", "interval_score_MW", None),
        ("hourly", "n", np.nan),
    ],
)
def test_invalid_required_interval_evidence_fails_before_write(
    tmp_path, monkeypatch, artifact, column, value
):
    daily_path, hourly_path = _write_report_inputs(tmp_path)
    path = daily_path if artifact == "daily" else hourly_path
    frame = pd.read_csv(path)
    if value is None:
        frame = frame.drop(columns=column)
    else:
        frame.loc[0, column] = value
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="interval evidence"):
        _run(tmp_path, monkeypatch)

    assert not (tmp_path / "report" / "generated_results.tex").exists()


@pytest.mark.parametrize("artifact", ["daily", "hourly"])
def test_non_retrospective_interval_evidence_fails_before_write(tmp_path, monkeypatch, artifact):
    daily_path, hourly_path = _write_report_inputs(tmp_path)
    path = daily_path if artifact == "daily" else hourly_path
    frame = pd.read_csv(path)
    frame["evaluation_period"] = "prospective_final"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="retrospective_final"):
        _run(tmp_path, monkeypatch)

    assert not (tmp_path / "report" / "generated_results.tex").exists()


def test_committed_managed_surfaces_pass_the_qualified_claim_gate():
    """Catch prohibited-claim regressions in authored prose before a full pipeline run.

    The gate splits on newlines, so a hard-wrapped sentence loses the qualifier that
    licenses its claim. This asserts every committed surface survives that split.
    """
    root = Path(__file__).resolve().parents[1]
    for relative in MANAGED_PRESENTATION_PATHS:
        text = (root / relative).read_text()
        before, after = presentation._parts(text, relative)
        presentation._check_qualified_claims(before + after, relative)
