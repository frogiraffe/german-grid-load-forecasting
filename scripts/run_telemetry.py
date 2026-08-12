"""Generate rolling error and interval telemetry from evaluated predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from loadfc.config import Config
from loadfc.evaluation.telemetry import rolling_backtest_telemetry

MODELS = [
    "SARIMAX",
    "xgboost",
    "lightgbm",
    "random_forest",
    "naive_1d",
    "seasonal_naive_7d",
    "ensemble",
]


def _join_intervals(
    predictions: pd.DataFrame,
    intervals: pd.DataFrame,
) -> pd.DataFrame:
    pd.testing.assert_index_equal(
        predictions.index,
        intervals.index,
        exact=True,
        check_names=True,
    )
    columns = ["lower_95", "upper_95"]
    selected = intervals[columns]
    if selected.isna().any().any() or not np.isfinite(selected.to_numpy(dtype="float64")).all():
        raise ValueError("interval bounds must be finite and complete")
    return predictions.join(selected, validate="one_to_one")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = Config.from_yaml(Path(args.config))
    results = cfg.path("results_dir")
    monitoring = cfg.monitoring
    rows = []

    for period, directory in (
        ("validation", results / "validation_predictions"),
        ("test", results / "predictions"),
    ):
        for model in MODELS:
            predictions = pd.read_csv(
                directory / f"{model}.csv",
                index_col="date",
                parse_dates=["date"],
            )
            lower = upper = None
            if period == "test":
                intervals = pd.read_csv(
                    results / "interval_predictions" / f"{model}.csv",
                    index_col="date",
                    parse_dates=["date"],
                )
                predictions = _join_intervals(predictions, intervals)
                lower, upper = "lower_95", "upper_95"
            telemetry = rolling_backtest_telemetry(
                predictions,
                window=int(monitoring["rolling_window"]),
                mape_warning=float(monitoring["mape_warning"]),
                coverage_floor=float(monitoring["coverage_floor"]),
                lower_column=lower,
                upper_column=upper,
            )
            telemetry.insert(0, "model", model)
            telemetry.insert(0, "period", period)
            rows.append(telemetry.reset_index())

    output = pd.concat(rows, ignore_index=True)
    metrics_dir = results / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(metrics_dir / "backtest_telemetry.csv", index=False)
    alerts = output[output["mape_alert"] | output["coverage_alert"]]
    alerts.to_csv(metrics_dir / "backtest_alerts.csv", index=False)
    print(f"wrote {len(output)} telemetry rows and {len(alerts)} alerts")


if __name__ == "__main__":
    main()
