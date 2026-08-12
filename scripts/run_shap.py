"""SHAP feature attribution for the tree models (XGBoost, LightGBM).

The comparison table reports error magnitudes but not feature contributions.
This fits each model on observations preceding the retrospective final period and
explains those previously inspected predictions with SHAP TreeExplainer. Outputs:

  results/analysis/shap_importance.csv   mean |SHAP| per feature, per model
  results/figs/14_shap_importance.png    global importance bar chart
  results/figs/15_shap_beeswarm.png      per-observation effects (XGBoost)

Usage:
    python scripts/run_shap.py [--config config.yaml]
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from loadfc.config import Config
from loadfc.features.assemble import build_features, exog_columns, feature_matrix
from loadfc.models.ml import make_estimator

warnings.simplefilter("ignore")

MODELS = ["xgboost", "lightgbm"]
_BLUE, _ORANGE = "#2c7fb8", "#e6820a"


def _mean_abs_shap(shap_values: np.ndarray, feature_names: list[str]) -> pd.Series:
    importance = np.abs(np.asarray(shap_values, dtype="float64")).mean(axis=0)
    return pd.Series(importance, index=feature_names).sort_values(ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = Config.from_yaml(Path(args.config))

    dataset = pd.read_parquet(cfg.path("processed_dir") / "dataset.parquet")
    feats = build_features(dataset, cfg)
    cols = exog_columns("ml")

    matrix = feature_matrix(feats, "ml")
    test_start = cfg.split.test_start
    train = matrix[matrix.index < test_start]
    test = matrix[(matrix.index >= test_start) & (matrix.index <= cfg.split.test_end)]
    X_train, y_train = train[cols], train["daily_load"]
    X_test = test[cols]

    importances: dict[str, pd.Series] = {}
    explanations: dict[str, shap.Explanation] = {}
    for kind in MODELS:
        params = dict(cfg.models[kind])
        params.setdefault("random_state", cfg.seed)
        est = make_estimator(kind, params)
        est.fit(X_train.to_numpy(dtype="float64"), y_train.to_numpy(dtype="float64"))

        explainer = shap.TreeExplainer(est)
        exp = explainer(X_test.to_numpy(dtype="float64"))
        exp.feature_names = cols
        explanations[kind] = exp
        importances[kind] = _mean_abs_shap(exp.values, cols)
        print(f"{kind}: top feature = {importances[kind].index[0]}")

    imp_df = pd.DataFrame(importances)
    imp_df["mean"] = imp_df.mean(axis=1)
    imp_df = imp_df.sort_values("mean", ascending=False)

    out = cfg.path("results_dir") / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    imp_df.to_csv(out / "shap_importance.csv", index_label="feature")
    print("shap_importance.csv written")

    figs = cfg.path("results_dir") / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    _importance_bar(imp_df, figs / "14_shap_importance.png")
    _beeswarm(explanations["xgboost"], figs / "15_shap_beeswarm.png")
    print("figures written to", figs)


def _importance_bar(imp_df: pd.DataFrame, path: Path) -> None:
    order = imp_df.sort_values("mean")
    y = range(len(order))
    fig, ax = plt.subplots(figsize=(9, 6))
    h = 0.4
    ax.barh([i + h / 2 for i in y], order["xgboost"], height=h, color=_BLUE, label="XGBoost")
    ax.barh([i - h / 2 for i in y], order["lightgbm"], height=h, color=_ORANGE, label="LightGBM")
    ax.set_yticks(list(y))
    ax.set_yticklabels(order.index)
    ax.set_xlabel("mean |SHAP| (MW)")
    ax.set_title("Global feature importance (mean absolute SHAP)")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _beeswarm(exp: shap.Explanation, path: Path) -> None:
    plt.figure()
    shap.plots.beeswarm(exp, max_display=12, show=False)
    plt.title("Per-observation SHAP effects — XGBoost (retrospective final period)")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=110)
    plt.close()


if __name__ == "__main__":
    main()
