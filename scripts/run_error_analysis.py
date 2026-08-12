"""Build temporal error slices and a validation-to-test stability view."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from loadfc.config import Config
from loadfc.evaluation.slices import temporal_error_slices

MODELS = [
    "ensemble",
    "SARIMAX",
    "xgboost",
    "lightgbm",
    "random_forest",
    "naive_1d",
    "seasonal_naive_7d",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = Config.from_yaml(Path(args.config))
    results = cfg.path("results_dir")
    metrics_dir = results / "metrics"

    rows: list[dict] = []
    for period, directory in [
        ("validation", results / "validation_predictions"),
        ("test", results / "predictions"),
    ]:
        for model in MODELS:
            frame = pd.read_csv(directory / f"{model}.csv", index_col=0, parse_dates=[0])
            rows.extend(temporal_error_slices(frame, model, period))
    slices = pd.DataFrame(rows)
    slices.to_csv(metrics_dir / "error_slices.csv", index=False)

    validation = pd.read_csv(metrics_dir / "validation_metrics.csv", index_col=0)
    test = pd.read_csv(metrics_dir / "test_metrics.csv", index_col=0)
    gap = pd.DataFrame(
        {
            "validation_MAPE": validation["MAPE"],
            "test_MAPE": test["MAPE"],
        }
    )
    gap["absolute_gap_pp"] = gap["test_MAPE"] - gap["validation_MAPE"]
    gap["relative_gap_pct"] = gap["absolute_gap_pp"] / gap["validation_MAPE"] * 100
    gap.to_csv(metrics_dir / "generalization_gap.csv", index_label="model")

    print("wrote error_slices.csv and generalization_gap.csv")


if __name__ == "__main__":
    main()
