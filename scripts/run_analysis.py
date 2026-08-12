"""Generate the report analyses from the committed dataset and write
them to results/analysis/: full descriptive statistics, VIF of the exogenous
design, the SARIMAX AIC grid search (top models) and the final-model exogenous
coefficients. These back Appendices A, C-F of the report.

Usage:
    python scripts/run_analysis.py [--config config.yaml]
"""

from __future__ import annotations

import argparse
import calendar
import itertools
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from loadfc.config import Config
from loadfc.evaluation.vif import compute_vif
from loadfc.features.assemble import build_features, exog_columns, feature_matrix

warnings.simplefilter("ignore")

_DESCRIPTIVE_SCOPE = "full_available_history_descriptive"
_TEMPERATURE_EDGES = list(range(-10, 36, 5))


def load_profile_weekday(feats: pd.DataFrame) -> pd.DataFrame:
    load = pd.to_numeric(feats["daily_load"], errors="coerce")
    valid = np.isfinite(load)
    frame = pd.DataFrame(
        {
            "weekday_order": pd.DatetimeIndex(feats.index).dayofweek + 1,
            "daily_load": load,
        },
        index=feats.index,
    )[valid]
    grouped = frame.groupby("weekday_order", sort=True)["daily_load"].agg(["mean", "size"])
    if grouped.index.tolist() != list(range(1, 8)) or (grouped["size"] == 0).any():
        raise ValueError("weekday profile requires all seven weekdays")
    return pd.DataFrame(
        {
            "weekday_order": grouped.index.astype(int),
            "weekday": [calendar.day_name[index - 1] for index in grouped.index],
            "mean_load_MW": grouped["mean"].to_numpy(dtype="float64"),
            "n_days": grouped["size"].to_numpy(dtype=int),
            "evidence_scope": _DESCRIPTIVE_SCOPE,
        }
    )


def load_profile_month(feats: pd.DataFrame) -> pd.DataFrame:
    load = pd.to_numeric(feats["daily_load"], errors="coerce")
    valid = np.isfinite(load)
    frame = pd.DataFrame(
        {
            "month_order": pd.DatetimeIndex(feats.index).month,
            "daily_load": load,
        },
        index=feats.index,
    )[valid]
    grouped = frame.groupby("month_order", sort=True)["daily_load"].agg(["mean", "size"])
    if grouped.index.tolist() != list(range(1, 13)) or (grouped["size"] == 0).any():
        raise ValueError("month profile requires all twelve months")
    return pd.DataFrame(
        {
            "month_order": grouped.index.astype(int),
            "month": [calendar.month_name[index] for index in grouped.index],
            "mean_load_MW": grouped["mean"].to_numpy(dtype="float64"),
            "n_days": grouped["size"].to_numpy(dtype=int),
            "evidence_scope": _DESCRIPTIVE_SCOPE,
        }
    )


def temperature_load_curve(feats: pd.DataFrame) -> pd.DataFrame:
    temperature = pd.to_numeric(feats["Temp_forecast"], errors="coerce")
    load = pd.to_numeric(feats["daily_load"], errors="coerce")
    eligible = np.isfinite(temperature) & np.isfinite(load)
    if not eligible.any():
        raise ValueError("temperature curve requires finite forecast-origin pairs")
    selected_temperature = temperature[eligible]
    if ((selected_temperature < _TEMPERATURE_EDGES[0]) | (selected_temperature >= _TEMPERATURE_EDGES[-1])).any():
        raise ValueError("eligible forecast-origin temperature outside [-10,35)")
    bins = pd.cut(
        selected_temperature,
        bins=_TEMPERATURE_EDGES,
        right=False,
        include_lowest=True,
    )
    grouped = pd.DataFrame({"bin": bins, "daily_load": load[eligible]}).groupby(
        "bin", observed=False, sort=True
    )["daily_load"].agg(["mean", "size"])
    if len(grouped) != 9 or (grouped["size"] == 0).any():
        raise ValueError("temperature curve requires all nine fixed bins")
    if int(grouped["size"].sum()) != int(eligible.sum()):
        raise ValueError("temperature curve counts disagree with eligible rows")
    return pd.DataFrame(
        {
            "bin_order": range(1, 10),
            "lower_C": _TEMPERATURE_EDGES[:-1],
            "upper_C": _TEMPERATURE_EDGES[1:],
            "mean_load_MW": grouped["mean"].to_numpy(dtype="float64"),
            "n_days": grouped["size"].to_numpy(dtype=int),
            "evidence_scope": _DESCRIPTIVE_SCOPE,
        }
    )


def descriptive_stats(feats: pd.DataFrame) -> pd.DataFrame:
    holiday_any = (
        feats[["holiday_national", "holiday_religious", "bridge_day"]].sum(axis=1) > 0
    ).astype(int)
    cols = {
        "daily_load (MW)": feats["daily_load"],
        "Temp_forecast (C)": feats["Temp_forecast"],
        "Wind_forecast (km/h)": feats["Wind_forecast"],
        "HDD (gun-C)": feats["HDD"],
        "CDD (gun-C)": feats["CDD"],
        "Weekend": feats["Weekend"],
        "Holiday_any": holiday_any,
        "L_t-1 (MW)": feats["L_t-1"],
        "L_t-7 (MW)": feats["L_t-7"],
    }
    rows = {}
    for name, s in cols.items():
        d = s.describe()
        rows[name] = {
            "n": int(d["count"]),
            "mean": d["mean"],
            "std": d["std"],
            "min": d["min"],
            "p25": d["25%"],
            "median": d["50%"],
            "p75": d["75%"],
            "max": d["max"],
            "skew": s.skew(),
        }
    return pd.DataFrame(rows).T


def vif_table(feats: pd.DataFrame, test_start) -> pd.DataFrame:
    train = feats[feats.index < test_start]
    sar = compute_vif(train[exog_columns("sarimax")].dropna())
    ml = compute_vif(train[exog_columns("ml")].dropna())
    out = pd.DataFrame({"VIF_SARIMAX": sar, "VIF_ML": ml})
    return out.sort_values("VIF_ML", ascending=False)


def _fit(endog, exog, order, seasonal_order, enforce=True):
    res = SARIMAX(
        endog,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=enforce,
        enforce_invertibility=enforce,
    ).fit(disp=False)
    return res


def grid_search(endog, exog, seasonal_s: int = 7) -> pd.DataFrame:
    # Order selection needs comparable, invertibility-constrained likelihoods:
    # without enforcement, MA models find non-invertible roots and inflate AIC.
    rows = []
    for p, q, P, Q in itertools.product((0, 1, 2), (0, 1, 2), (0, 1), (0, 1)):
        order, seasonal = (p, 1, q), (P, 0, Q, seasonal_s)
        try:
            res = _fit(endog, exog, order, seasonal, enforce=True)
        except Exception:
            continue
        rows.append(
            {
                "model": f"SARIMA({p},1,{q})({P},0,{Q}){seasonal_s}",
                "AIC": res.aic,
                "BIC": res.bic,
            }
        )
    return pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)


def sarimax_coeffs(endog, exog_df, order, seasonal_order) -> pd.DataFrame:
    # Match the evaluation forecaster, which runs unconstrained.
    res = _fit(endog, exog_df, order, seasonal_order, enforce=False)
    names = list(exog_df.columns)
    coef = res.params[: len(names)]
    se = res.bse[: len(names)]
    pval = res.pvalues[: len(names)]
    return pd.DataFrame(
        {"coefficient_MW": coef.to_numpy(), "std_error": se.to_numpy(), "p_value": pval.to_numpy()},
        index=names,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = Config.from_yaml(Path(args.config))

    dataset = pd.read_parquet(cfg.path("processed_dir") / "dataset.parquet")
    feats = build_features(dataset, cfg)
    out = cfg.path("results_dir") / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    load_profile_weekday(feats).to_csv(out / "load_profile_weekday.csv", index=False)
    load_profile_month(feats).to_csv(out / "load_profile_month.csv", index=False)
    temperature_load_curve(feats).to_csv(out / "temperature_load_curve.csv", index=False)

    descriptive_stats(feats).to_csv(out / "descriptive_stats.csv")
    print("descriptive_stats.csv written")

    vif_table(feats, cfg.split.test_start).to_csv(out / "vif.csv")
    print("vif.csv written")

    # Select the order on training data; estimate reported coefficients on data
    # available before the July test.
    sar = feature_matrix(feats, "sarimax")
    validation_start = cfg.split.train_end + timedelta(days=1)
    selection_df = sar[sar.index < validation_start]

    grid = grid_search(
        selection_df["daily_load"].to_numpy(dtype="float64"),
        selection_df[exog_columns("sarimax")].to_numpy(dtype="float64"),
    )
    grid.head(10).to_csv(out / "sarimax_grid.csv", index_label="rank")
    print("sarimax_grid.csv written; best:", grid.iloc[0]["model"])

    fit_df = sar[sar.index < cfg.split.test_start]
    endog = fit_df["daily_load"].to_numpy(dtype="float64")
    exog_df = fit_df[exog_columns("sarimax")].astype("float64")
    s = cfg.models["sarimax"]
    coeffs = sarimax_coeffs(endog, exog_df, tuple(s["order"]), tuple(s["seasonal_order"]))
    coeffs.to_csv(out / "sarimax_coeffs.csv")
    print("sarimax_coeffs.csv written")
    print("saved analysis to", out)


if __name__ == "__main__":
    main()
