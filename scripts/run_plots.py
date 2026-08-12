"""Generate every figure into results/figs/ from the committed data and results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from loadfc.config import Config
from loadfc.features.assemble import build_features
from loadfc.viz import plots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = Config.from_yaml(Path(args.config))

    proc = cfg.path("processed_dir")
    dataset = pd.read_parquet(proc / "dataset.parquet")
    feats = build_features(dataset, cfg)
    load = dataset["daily_load"]
    figs = cfg.path("results_dir") / "figs"

    plots.timeseries(load, cfg.split.test_start, figs / "01_timeseries.png")
    plots.stl_decomposition(load, figs / "02_stl.png")
    plots.monthly_violin(load, figs / "03_monthly_violin.png")
    plots.weekly_profile(load, figs / "04_weekly_profile.png")
    plots.acf_pacf(load, figs / "05_acf_pacf.png")
    plots.temp_load(dataset["Temp_t"], load, figs / "06_temp_load.png")
    plots.correlation(feats, figs / "07_correlation.png")

    metrics_dir = cfg.path("results_dir") / "metrics"
    preds_dir = cfg.path("results_dir") / "predictions"
    for name in [
        "SARIMAX",
        "xgboost",
        "lightgbm",
        "random_forest",
        "naive_1d",
        "seasonal_naive_7d",
        "ensemble",
    ]:
        p = pd.read_csv(preds_dir / f"{name}.csv", index_col=0, parse_dates=[0])
        plots.forecast_vs_actual(p["actual"], p["forecast"], name, figs / f"08_forecast_{name}.png")

    sarimax = pd.read_csv(preds_dir / "SARIMAX.csv", index_col=0, parse_dates=[0])
    plots.sarimax_residuals(
        sarimax["actual"] - sarimax["forecast"], figs / "08_sarimax_residuals.png"
    )

    metrics_df = pd.read_csv(metrics_dir / "test_metrics.csv", index_col=0)
    plots.model_comparison(metrics_df, figs / "09_model_comparison.png")
    plots.dm_heatmap(
        pd.read_csv(metrics_dir / "diebold_mariano.csv", index_col=0),
        figs / "10_diebold_mariano.png",
    )
    plots.rolling_origin(
        pd.read_csv(metrics_dir / "rolling_origin_mape.csv", index_col=0),
        figs / "11_rolling_origin.png",
    )

    print("figures written to", figs)
    print("count:", len(list(figs.glob("*.png"))))


if __name__ == "__main__":
    main()
