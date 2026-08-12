"""Render deterministic LaTeX tables from the committed result files."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from loadfc.config import Config
from loadfc.presentation import (
    MANAGED_PRESENTATION_PATHS,
    PresentationValues,
    check_managed_presentation,
    load_presentation_values,
    model_display_name,
    replace_managed_presentation,
)


def _name(value: str) -> str:
    return model_display_name(value)


def _latex(value: object) -> str:
    return str(value).replace("%", r"\%").replace("_", r"\_")


def _metrics_table(frame: pd.DataFrame) -> str:
    rows = [
        f"{_name(model)} & {row.RMSE:.1f} & {row.MAE:.1f} & {row.MAPE:.3f} & {row.MASE:.3f} \\\\"
        for model, row in frame.iterrows()
    ]
    return "\n".join(rows)


def _rolling_table(frame: pd.DataFrame) -> str:
    rows = []
    for window, row in frame.iterrows():
        rows.append(
            f"{window} & {row['SARIMAX']:.3f} & {row['xgboost']:.3f} & "
            f"{row['lightgbm']:.3f} & {row['random_forest']:.3f} & "
            f"{row['seasonal_naive_7d']:.3f} \\\\"
        )
    return "\n".join(rows)


def _interval_table(frame: pd.DataFrame) -> str:
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            f"{_name(row['model'])} & {_latex(str(row['method']).title())} & "
            f"{_latex(row['level'])} & "
            f"{row['nominal']:.2f} & "
            f"{row['empirical_coverage']:.3f} & {row['mean_width_MW']:.1f} & "
            f"{row['interval_score_MW']:.1f} & {int(row['n'])} \\\\"
        )
    return "\n".join(rows)


def _validate_interval_evidence(
    frame: pd.DataFrame,
    *,
    artifact: str,
    width_column: str,
    score_column: str,
    methods: set[str],
) -> None:
    required = {
        "method",
        "level",
        "evaluation_period",
        "coverage_scope",
        "nominal",
        "empirical_coverage",
        width_column,
        score_column,
        "n",
    }
    missing = required - set(frame)
    if missing or frame.empty:
        raise ValueError(f"{artifact} interval evidence is missing required fields: {sorted(missing)}")
    if not frame["evaluation_period"].eq("retrospective_final").all():
        raise ValueError(f"{artifact} interval evidence must use retrospective_final")
    if not methods <= set(frame["method"]):
        raise ValueError(f"{artifact} interval evidence is missing required methods")

    expected_scopes = {
        "adaptive": "prequential_monitoring_no_unconditional_time_series_coverage"
    }
    for method, scope in frame[["method", "coverage_scope"]].itertuples(index=False):
        expected = expected_scopes.get(str(method), "empirical_retrospective")
        if scope != expected:
            raise ValueError(f"{artifact} interval evidence has an invalid coverage scope")

    numeric_columns = ("nominal", "empirical_coverage", width_column, score_column, "n")
    for column in numeric_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not values.map(math.isfinite).all():
            raise ValueError(f"{artifact} interval evidence has non-finite {column}")
    if not pd.to_numeric(frame["nominal"]).between(0.0, 1.0, inclusive="neither").all():
        raise ValueError(f"{artifact} interval evidence has invalid nominal levels")
    if not pd.to_numeric(frame["empirical_coverage"]).between(0.0, 1.0).all():
        raise ValueError(f"{artifact} interval evidence has invalid empirical coverage")
    if not (pd.to_numeric(frame[[width_column, score_column]].stack()) >= 0.0).all():
        raise ValueError(f"{artifact} interval evidence has negative width or interval score")
    n = pd.to_numeric(frame["n"])
    if not ((n > 0) & (n % 1 == 0)).all():
        raise ValueError(f"{artifact} interval evidence has invalid n")


def _weather_table(frame: pd.DataFrame) -> str:
    rows = []
    for model, row in frame.iterrows():
        if model in {"naive_1d", "seasonal_naive_7d"}:
            continue
        rows.append(
            f"{_name(model)} & {row['operational_MAPE']:.3f} & "
            f"{row['persistence_MAPE']:.3f} & {row['delta_pp']:+.3f} \\\\"
        )
    return "\n".join(rows)


def _gap_table(frame: pd.DataFrame) -> str:
    rows = []
    for model, row in frame.iterrows():
        rows.append(
            f"{_name(model)} & {row['validation_MAPE']:.3f} & "
            f"{row['test_MAPE']:.3f} & {row['absolute_gap_pp']:+.3f} \\\\"
        )
    return "\n".join(rows)


def _render_report_data(cfg: Config, values: PresentationValues) -> str:
    metrics_dir = cfg.path("results_dir") / "metrics"
    validation = pd.read_csv(metrics_dir / "validation_metrics.csv", index_col=0)
    calibration = pd.read_csv(metrics_dir / "calibration_metrics.csv", index_col=0)
    test = pd.read_csv(metrics_dir / "test_metrics.csv", index_col=0)
    rolling = pd.read_csv(metrics_dir / "rolling_origin_mape.csv", index_col=0)
    intervals = pd.read_csv(metrics_dir / "interval_coverage.csv")
    _validate_interval_evidence(
        intervals,
        artifact="daily",
        width_column="mean_width_MW",
        score_column="interval_score_MW",
        methods={"fixed", "adaptive"},
    )
    intervals = intervals[intervals["level"] == "95%"]
    gap = pd.read_csv(metrics_dir / "generalization_gap.csv", index_col=0)
    weather = pd.read_csv(metrics_dir / "weather_ablation.csv", index_col=0)

    selected_hourly = values.hourly_model
    anchor_model = values.daily_anchor_model
    reconciliation_gain = (1.0 - values.hourly_reconciled_mae / values.hourly_raw_mae) * 100.0

    best_model = values.daily.model
    best = values.daily
    ensemble = test.loc["ensemble"]
    interval_values = {item.method: item for item in values.hourly_intervals}
    symmetric_interval = interval_values["symmetric"]
    adaptive_interval = interval_values["adaptive"]
    cqr_interval = interval_values["cqr"]
    text = f"""% Generated by scripts/render_report_data.py. Do not edit by hand.
\\newcommand{{\\BestModel}}{{{_name(best_model)}}}
\\newcommand{{\\BestMAPE}}{{{best.mape:.3f}\\%}}
\\newcommand{{\\BestMASE}}{{{best.mase:.3f}}}
\\newcommand{{\\BestMAE}}{{{best.mae:.1f}}}
\\newcommand{{\\FinalRole}}{{{_latex(values.final_role)}}}
\\newcommand{{\\FinalRoleLabel}}{{{_latex(values.final_role_label)}}}
\\newcommand{{\\FinalStart}}{{{values.final_start}}}
\\newcommand{{\\FinalEnd}}{{{values.final_end}}}
\\newcommand{{\\FinalN}}{{{values.daily.n}}}
\\newcommand{{\\SourceRevision}}{{{_latex(values.source_revision)}}}
\\newcommand{{\\BundleFingerprint}}{{{values.bundle_fingerprint}}}
\\newcommand{{\\DailyProtocolFingerprint}}{{{_latex(values.daily_protocol_fingerprint)}}}
\\newcommand{{\\HourlyProtocolFingerprint}}{{{_latex(values.hourly_protocol_fingerprint)}}}
\\newcommand{{\\EnsembleMAPE}}{{{ensemble["MAPE"]:.3f}\\%}}
\\newcommand{{\\EnsembleMASE}}{{{ensemble["MASE"]:.3f}}}
\\newcommand{{\\EnsembleMAE}}{{{ensemble["MAE"]:.1f}}}
\\newcommand{{\\HourlySelectedModel}}{{{_name(selected_hourly)}}}
\\newcommand{{\\DailyAnchorModel}}{{{_name(anchor_model)}}}
\\newcommand{{\\HourlyValidationSelectedMAE}}{{{values.hourly_validation_mae:.1f}}}
\\newcommand{{\\HourlyValidationLightGBMMAE}}{{{values.hourly_candidate_validation_mae:.1f}}}
\\newcommand{{\\HourlyTestSelectedMAE}}{{{values.hourly_reconciled_mae:.1f}}}
\\newcommand{{\\HourlyTestLightGBMMAE}}{{{values.hourly_candidate_final_mae:.1f}}}
\\newcommand{{\\BootstrapDifference}}{{{values.bootstrap_difference:.1f}}}
\\newcommand{{\\BootstrapLower}}{{{values.bootstrap_lower:.1f}}}
\\newcommand{{\\BootstrapUpper}}{{{values.bootstrap_upper:.1f}}}
\\newcommand{{\\BootstrapProbability}}{{{100.0 * values.bootstrap_probability:.2f}\\%}}
\\newcommand{{\\BootstrapN}}{{{values.bootstrap_n_days}}}
\\newcommand{{\\RawHourlyMAE}}{{{values.hourly_raw_mae:.1f}}}
\\newcommand{{\\ReconciledHourlyMAE}}{{{values.hourly_reconciled_mae:.1f}}}
\\newcommand{{\\HourlyN}}{{{values.hourly_n}}}
\\newcommand{{\\ReconciliationGain}}{{{reconciliation_gain:.2f}\\%}}
\\newcommand{{\\SymmetricNominalLevel}}{{{_latex(symmetric_interval.level)}}}
\\newcommand{{\\SymmetricCoverage}}{{{100.0 * symmetric_interval.empirical_coverage:.2f}\\%}}
\\newcommand{{\\SymmetricWidth}}{{{symmetric_interval.mean_width:.1f}}}
\\newcommand{{\\SymmetricIntervalScore}}{{{symmetric_interval.interval_score:.1f}}}
\\newcommand{{\\SymmetricN}}{{{symmetric_interval.n}}}
\\newcommand{{\\AdaptiveNominalLevel}}{{{_latex(adaptive_interval.level)}}}
\\newcommand{{\\AdaptiveCoverage}}{{{100.0 * adaptive_interval.empirical_coverage:.2f}\\%}}
\\newcommand{{\\AdaptiveWidth}}{{{adaptive_interval.mean_width:.1f}}}
\\newcommand{{\\AdaptiveIntervalScore}}{{{adaptive_interval.interval_score:.1f}}}
\\newcommand{{\\AdaptiveN}}{{{adaptive_interval.n}}}
\\newcommand{{\\CQRNominalLevel}}{{{_latex(cqr_interval.level)}}}
\\newcommand{{\\CQRCoverage}}{{{100.0 * cqr_interval.empirical_coverage:.2f}\\%}}
\\newcommand{{\\CQRWidth}}{{{cqr_interval.mean_width:.1f}}}
\\newcommand{{\\CQRIntervalScore}}{{{cqr_interval.interval_score:.1f}}}
\\newcommand{{\\CQRN}}{{{cqr_interval.n}}}
\\newcommand{{\\EmpiricalRetrospectiveCoverageLabel}}{{Empirical retrospective coverage.}}
\\newcommand{{\\PrequentialMonitoringCoverageLabel}}{{Prequential monitoring evidence; no unconditional time-series coverage guarantee.}}
\\newcommand{{\\ArtifactReplayError}}{{{7.275957614183426e-12:.2e}}}

\\newcommand{{\\ValidationRows}}{{%
{_metrics_table(validation)}
}}

\\newcommand{{\\CalibrationRows}}{{%
{_metrics_table(calibration)}
}}

\\newcommand{{\\TestRows}}{{%
{_metrics_table(test)}
}}

\\newcommand{{\\RollingRows}}{{%
{_rolling_table(rolling)}
}}

\\newcommand{{\\IntervalRows}}{{%
{_interval_table(intervals)}
}}

\\newcommand{{\\WeatherRows}}{{%
{_weather_table(weather)}
}}

\\newcommand{{\\GeneralizationRows}}{{%
{_gap_table(gap)}
}}
"""
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--check", action="store_true", help="report drift without writing")
    args = parser.parse_args()
    cfg = Config.from_yaml(Path(args.config))
    values = load_presentation_values(cfg.path("results_dir"))
    text = _render_report_data(cfg, values)
    destination = cfg.root / "report" / "generated_results.tex"

    documents: dict[Path, str] = {}
    for relative in MANAGED_PRESENTATION_PATHS:
        path = cfg.root / relative
        if not path.is_file():
            raise ValueError(f"missing managed presentation surface: {path}")
        current = path.read_text(encoding="utf-8")
        documents[path] = replace_managed_presentation(current, values, relative)

    if args.check:
        drift = []
        if not destination.is_file() or destination.read_text(encoding="utf-8") != text:
            drift.append(str(destination))
        for relative in MANAGED_PRESENTATION_PATHS:
            path = cfg.root / relative
            try:
                check_managed_presentation(path.read_text(encoding="utf-8"), values, relative)
            except ValueError:
                drift.append(str(path))
        if drift:
            raise ValueError(f"presentation drift: {', '.join(drift)}")
        print("presentation current")
        return

    destination.write_text(text, encoding="utf-8")
    for path, updated in documents.items():
        path.write_text(updated, encoding="utf-8")
    print("wrote", destination, "and", len(documents), "managed presentation blocks")


if __name__ == "__main__":
    main()
