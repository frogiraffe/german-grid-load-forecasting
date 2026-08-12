"""Tune the tree-based forecasters with Optuna and report the best hyperparameters.

The study runs only on the training period (an expanding-window
TimeSeriesSplit ending before the 2024 validation period) and minimises
cross-validated MASE --
the seasonal-naive-scaled error the study optimises. It writes the winners to
results/tuning/best_params.json and prints them next to the committed config.yaml
values for comparison.

This tuning example is not part of the evaluation pipeline.
It does NOT overwrite config.yaml. The hyperparameters in config.yaml are the frozen
baseline behind every number in the README; a fresh study is stochastic and can
land on slightly different values. To adopt new values, copy them into
config.yaml by hand and re-run run_evaluate.py / run_plots.py.

Usage:
    python scripts/run_tune.py [--config config.yaml] [--models xgboost lightgbm random_forest]
                               [--n-trials N] [--cv-splits K]
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import timedelta
from pathlib import Path

import pandas as pd

from loadfc.config import Config
from loadfc.features.assemble import build_features, exog_columns, feature_matrix
from loadfc.tuning.study import training_slice, tune_model

warnings.simplefilter("ignore")

ML_MODELS = ["xgboost", "lightgbm", "random_forest"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--models", nargs="+", default=ML_MODELS, choices=ML_MODELS)
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--cv-splits", type=int, default=None)
    args = parser.parse_args()

    cfg = Config.from_yaml(Path(args.config))
    n_trials = args.n_trials or cfg.tuning["n_trials"]
    cv_splits = args.cv_splits or cfg.tuning["cv_splits"]

    dataset = pd.read_parquet(cfg.path("processed_dir") / "dataset.parquet")
    feats = build_features(dataset, cfg)
    cols = exog_columns("ml")
    validation_start = cfg.split.train_end + timedelta(days=1)
    matrix = training_slice(feature_matrix(feats, "ml"), validation_start)
    print(
        f"tuning on {len(matrix)} rows before {validation_start} "
        f"({n_trials} trials, {cv_splits}-fold time-series CV)\n"
    )

    out: dict[str, dict] = {}
    for kind in args.models:
        result = tune_model(
            kind, matrix, cols, n_trials=n_trials, n_splits=cv_splits, seed=cfg.seed
        )
        out[kind] = {"cv_mase": result.value, "params": result.params}
        committed = cfg.models[kind]
        print(f"[{kind}] best CV MASE = {result.value:.4f}")
        for name, value in result.params.items():
            base = committed.get(name)
            fmt = f"{value:.4g}" if isinstance(value, float) else str(value)
            print(f"    {name:18s} tuned={fmt:<10} config={base}")
        print()

    results_dir = cfg.path("results_dir") / "tuning"
    results_dir.mkdir(parents=True, exist_ok=True)
    dest = results_dir / "best_params.json"
    dest.write_text(json.dumps(out, indent=2))
    print("wrote", dest)
    print("config.yaml is unchanged — copy values in by hand to adopt them.")


if __name__ == "__main__":
    main()
