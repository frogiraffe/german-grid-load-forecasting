"""Build temporal error slices and a validation-to-test stability view."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
DISPLAY = {
    "ensemble": "Ensemble",
    "xgboost": "XGBoost",
    "seasonal_naive_7d": "Seasonal naive",
}


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

    _plot(slices, gap, results / "figs" / "17_temporal_stability.png")
    print("wrote error_slices.csv, generalization_gap.csv and temporal stability figure")


def _plot(slices: pd.DataFrame, gap: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    gap.loc[MODELS, ["validation_MAPE", "test_MAPE"]].rename(
        index=lambda value: DISPLAY.get(value, value)
    ).plot(kind="bar", ax=axes[0], color=["#7fcdbb", "#2c7fb8"])
    axes[0].set(title="Validation-to-test stability", ylabel="MAPE (%)", xlabel="")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].legend(["Validation", "July 2026 test"])

    monthly = slices[
        (slices["period"] == "test")
        & (slices["slice_type"] == "month")
        & slices["model"].isin(DISPLAY)
    ]
    for model, group in monthly.groupby("model"):
        group = group.sort_values("slice")
        axes[1].plot(
            group["slice"],
            group["MAPE"],
            marker="o",
            label=DISPLAY[model],
        )
    axes[1].set(title="Test-period error", ylabel="MAPE (%)", xlabel="")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].legend()
    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
