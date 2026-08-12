"""Fixed and adaptive conformal intervals for the retrospective-final block.

Residuals from the calibration period initialise both methods. Adaptive
intervals are updated only after each test-period actual becomes available.
Because time-series residuals are not exchangeable, all coverage is reported as
empirical retrospective evidence. Adaptive evidence is prequential monitoring,
not an unconditional time-series coverage guarantee.

Usage:
    python scripts/run_intervals.py [--config config.yaml] [--calib 90]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from loadfc.config import Config
from loadfc.evaluation.conformal import (
    adaptive_conformal_interval,
    interval_evidence,
    split_conformal_interval,
)
from loadfc.evaluation.protocol import assert_compatible_artifacts, protocol_fingerprint
from loadfc.evaluation.provenance import artifact_results_root
from loadfc.viz import plots

MODELS = [
    "SARIMAX",
    "xgboost",
    "lightgbm",
    "random_forest",
    "naive_1d",
    "seasonal_naive_7d",
    "ensemble",
]
LEVELS = {"80%": 0.20, "95%": 0.05}
_PROVENANCE_COLUMNS = [
    "forecast_origin",
    "valid_time",
    "weather_source_run",
    "weather_availability_assumption",
]
_PROTOCOL_COLUMNS = [
    "stream_id",
    "protocol_fingerprint",
    "evaluation_period",
    "point_state_policy",
    "interval_state_policy",
]


def _read_prediction_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=[0], engine="python")


def interval_prediction_frame(test: pd.DataFrame) -> pd.DataFrame:
    """Preserve the point-prediction audit fields while conformal values are added."""

    required = {"actual", "forecast", *_PROVENANCE_COLUMNS, *_PROTOCOL_COLUMNS}
    missing = required - set(test)
    if missing:
        raise ValueError(f"test predictions are missing provenance columns: {sorted(missing)}")
    return test[["actual", "forecast", *_PROVENANCE_COLUMNS, *_PROTOCOL_COLUMNS]].copy()


def _artifact_metadata(frame: pd.DataFrame) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for column in _PROTOCOL_COLUMNS:
        if column not in frame:
            raise ValueError(f"prediction artifact missing protocol column: {column}")
        values = frame[column].dropna().unique()
        if len(values) != 1:
            raise ValueError(f"prediction artifact must have one {column}")
        metadata[column] = values[0]
    return metadata


def _assert_registered_stream(
    calibration: pd.DataFrame,
    retrospective_final: pd.DataFrame,
    manifest_path: Path,
) -> dict[str, object]:
    calibration_metadata = _artifact_metadata(calibration)
    final_metadata = _artifact_metadata(retrospective_final)
    assert_compatible_artifacts(calibration_metadata, final_metadata)

    manifest = json.loads(manifest_path.read_text())
    records = manifest.get("records") if isinstance(manifest, dict) else None
    stream_id = calibration_metadata["stream_id"]
    if not isinstance(records, dict) or stream_id not in records:
        raise ValueError("protocol manifest has no registered stream")
    record = records[stream_id]
    if not isinstance(record, dict):
        raise ValueError("protocol manifest record must be a mapping")
    if calibration_metadata["protocol_fingerprint"] != protocol_fingerprint(record):
        raise ValueError("protocol_fingerprint does not match registered stream")
    for field in ("point_state_policy", "interval_state_policy"):
        expected = json.dumps(record[field], sort_keys=True)
        if calibration_metadata[field] != expected:
            raise ValueError(f"{field} does not match registered stream")
    return final_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--calib", type=int, default=None, help="calibration days")
    args = parser.parse_args()
    cfg = Config.from_yaml(Path(args.config))

    results_dir = artifact_results_root(cfg)
    preds_dir = results_dir / "predictions"
    calibration_dir = results_dir / "calibration_predictions"
    metrics_dir = results_dir / "metrics"
    figs = results_dir / "figs"
    interval_dir = results_dir / "interval_predictions"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    interval_dir.mkdir(parents=True, exist_ok=True)
    calibration_days = args.calib or int(cfg.uncertainty["calibration_days"])
    gamma = float(cfg.uncertainty["adaptive_gamma"])
    window = int(cfg.uncertainty["adaptive_window"])

    rows: list[dict] = []
    fig_done = False
    for name in MODELS:
        calibration = _read_prediction_csv(calibration_dir / f"{name}.csv").iloc[
            -calibration_days:
        ]
        test = _read_prediction_csv(preds_dir / f"{name}.csv")
        final_metadata = _assert_registered_stream(
            calibration,
            test,
            results_dir / "evaluation_protocol.json",
        )
        interval_predictions = interval_prediction_frame(test)
        for label, alpha in LEVELS.items():
            fixed_lower, fixed_upper = split_conformal_interval(
                calibration["actual"], calibration["forecast"], test["forecast"], alpha
            )
            adaptive_lower, adaptive_upper, alpha_history = adaptive_conformal_interval(
                calibration["actual"],
                calibration["forecast"],
                test["actual"],
                test["forecast"],
                alpha,
                gamma=gamma,
                window=window,
            )
            suffix = label.removesuffix("%")
            interval_predictions[f"fixed_lower_{suffix}"] = fixed_lower
            interval_predictions[f"fixed_upper_{suffix}"] = fixed_upper
            interval_predictions[f"lower_{suffix}"] = adaptive_lower
            interval_predictions[f"upper_{suffix}"] = adaptive_upper
            interval_predictions[f"adaptive_alpha_{suffix}"] = alpha_history
            for method, lower, upper in [
                ("fixed", fixed_lower, fixed_upper),
                ("adaptive", adaptive_lower, adaptive_upper),
            ]:
                evidence = interval_evidence(test["actual"], lower, upper, alpha)
                row = {
                    "model": name,
                    "method": method,
                    "level": label,
                    "evaluation_period": final_metadata["evaluation_period"],
                    "stream_id": final_metadata["stream_id"],
                    "protocol_fingerprint": final_metadata["protocol_fingerprint"],
                    "point_state_policy": final_metadata["point_state_policy"],
                    "interval_state_policy": final_metadata["interval_state_policy"],
                    "coverage_scope": (
                        "empirical_retrospective"
                        if method == "fixed"
                        else "prequential_monitoring_no_unconditional_time_series_coverage"
                    ),
                    "nominal": evidence["nominal"],
                    "empirical_coverage": round(float(evidence["empirical_coverage"]), 4),
                    "mean_width_MW": round(float(evidence["mean_width"]), 1),
                    "interval_score_MW": round(float(evidence["interval_score"]), 1),
                    "n": evidence["n"],
                }
                rows.append(row)
            if name == "SARIMAX" and label == "95%" and not fig_done:
                plots.prediction_intervals(
                    test["actual"],
                    test["forecast"],
                    adaptive_lower,
                    adaptive_upper,
                    "SARIMAX adaptive conformal",
                    figs / "13_prediction_intervals.png",
                )
                fig_done = True
        # Keep canonical precision: actual/forecast must stay byte-identical to the
        # matching predictions/<model>.csv rows that downstream evidence compares against.
        interval_predictions.to_csv(interval_dir / f"{name}.csv")

    out = pd.DataFrame(rows)
    out.to_csv(metrics_dir / "interval_coverage.csv", index=False)
    print(out.to_string(index=False))
    print("\nsaved coverage to", metrics_dir / "interval_coverage.csv")
    print("saved figure to", figs / "13_prediction_intervals.png")


if __name__ == "__main__":
    main()
