"""Figure functions for the load-forecasting study. Each saves one PNG."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})
_BLUE = "#2c7fb8"


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def timeseries(load: pd.Series, test_start: date, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(load.index, load.values, color=_BLUE, lw=0.6)
    ax.axvline(test_start, color="crimson", ls="--", lw=1, label="retrospective final starts")  # type: ignore[arg-type]
    ax.set(
        title=f"Germany daily average grid load ({min(load.index).year}-{max(load.index).year})",
        xlabel="Date",
        ylabel="Load (MW)",
    )
    ax.legend()
    _save(fig, path)
def stl_decomposition(load: pd.Series, path: Path) -> None:
    res = STL(pd.Series(load.values, index=pd.to_datetime(load.index)), period=7, robust=True).fit()
    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    for ax, comp, name in zip(
        axes,
        [res.observed, res.trend, res.seasonal, res.resid],
        ["Observed", "Trend", "Seasonal (weekly)", "Residual"],
        strict=True,
    ):
        ax.plot(comp.index, comp.values, color=_BLUE, lw=0.6)
        ax.set_ylabel(name)
    axes[0].set_title("STL decomposition (period = 7)")
    _save(fig, path)


def monthly_violin(load: pd.Series, path: Path) -> None:
    df = pd.DataFrame({"load": load.values, "month": [d.month for d in load.index]})
    fig, ax = plt.subplots(figsize=(11, 4))
    sns.violinplot(data=df, x="month", y="load", color=_BLUE, ax=ax, inner="box")
    ax.set(title="Monthly distribution of daily load", xlabel="Month", ylabel="Load (MW)")
    _save(fig, path)


def weekly_profile(load: pd.Series, path: Path) -> None:
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    df = pd.DataFrame({"load": load.values, "dow": [d.weekday() for d in load.index]})
    g = df.groupby("dow")["load"].agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(days, g["mean"], yerr=g["std"], color=_BLUE, capsize=4)
    ax.set(title="Average load by day of week", ylabel="Load (MW)")
    _save(fig, path)


def acf_pacf(load: pd.Series, path: Path, lags: int = 60) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 6))
    plot_acf(load.values, lags=lags, ax=axes[0])
    plot_pacf(load.values, lags=lags, ax=axes[1], method="ywm")
    axes[0].set_title("ACF - daily load")
    axes[1].set_title("PACF - daily load")
    _save(fig, path)


def temp_load(temp: pd.Series, load: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    hb = ax.hexbin(temp.values, load.values, gridsize=30, cmap="Blues", mincnt=1)
    r = np.corrcoef(temp.values, load.values)[0, 1]
    ax.set(title=f"Temperature vs load (r = {r:.2f})", xlabel="Temperature (C)", ylabel="Load (MW)")
    fig.colorbar(hb, ax=ax, label="count")
    _save(fig, path)


def correlation(feats: pd.DataFrame, path: Path) -> None:
    cols = [
        "daily_load",
        "Temp_forecast",
        "Wind_forecast",
        "Weekend",
        "HDD",
        "CDD",
        "L_t-1",
        "L_t-7",
    ]
    corr = feats[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Feature correlation matrix")
    _save(fig, path)


def forecast_vs_actual(actual: pd.Series, forecast: pd.Series, name: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(actual.index, actual.values, color="black", lw=0.7, label="actual")
    ax.plot(forecast.index, forecast.values, color=_BLUE, lw=0.7, label=f"{name} forecast")
    start, end = pd.Timestamp(actual.index.min()), pd.Timestamp(actual.index.max())
    ax.set(
        title=f"{name} - retrospective final forecast vs actual ({start:%Y-%m-%d} to {end:%Y-%m-%d})",
        xlabel="Date",
        ylabel="Load (MW)",
    )
    ax.legend()
    _save(fig, path)


def sarimax_residuals(resid: pd.Series, path: Path, acf_lags: int = 30) -> None:
    """Four-panel residual diagnostics: series, histogram, Q-Q and ACF."""
    from scipy import stats

    r = np.asarray(resid, dtype="float64")
    if r.size < 2:
        raise ValueError("residual diagnostics require at least two observations")
    acf_lags = min(acf_lags, r.size - 1)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot(resid.index, r, color=_BLUE, lw=0.6)
    axes[0, 0].axhline(0, color="crimson", ls="--", lw=0.8)
    axes[0, 0].set(title="Residuals (time series)", xlabel="Date", ylabel="Residual (MW)")

    axes[0, 1].hist(r, bins=40, density=True, color=_BLUE, alpha=0.7)
    xs = np.linspace(r.min(), r.max(), 200)
    axes[0, 1].plot(xs, stats.norm.pdf(xs, r.mean(), r.std()), color="darkorange", lw=1.5)
    axes[0, 1].set(title="Residual distribution", xlabel="Residual (MW)")

    stats.probplot(r, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q plot (normal)")

    plot_acf(r, lags=acf_lags, ax=axes[1, 1], color=_BLUE, vlines_kwargs={"colors": _BLUE})
    axes[1, 1].set(title="Residual ACF", xlabel="Lag (days)")
    fig.suptitle("SARIMAX residual diagnostics", y=1.01)
    _save(fig, path)


def model_comparison(metrics_df: pd.DataFrame, path: Path) -> None:
    norm = metrics_df.apply(lambda c: (c - c.min()) / (c.max() - c.min()), axis=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(
        norm,
        annot=metrics_df.round(2).to_numpy(),
        fmt="",
        cmap="Greens_r",
        ax=ax,
        cbar_kws={"label": "column rank (0=best)"},
    )
    ax.set_title("Model comparison - retrospective final metrics")
    _save(fig, path)


def dm_heatmap(dm_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(dm_df, annot=True, fmt=".3f", cmap="RdBu_r", vmin=0, vmax=1, ax=ax)
    ax.set_title("Diebold-Mariano Bonferroni-adjusted p-values")
    _save(fig, path)


def rolling_origin(ro_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    ro_df.plot(kind="bar", ax=ax)
    ax.set(title="Rolling-origin MAPE by six-month window", xlabel="Window", ylabel="MAPE (%)")
    ax.legend(title="model")
    _save(fig, path)


def prediction_intervals(
    actual, forecast, lower, upper, name: str, path: Path, level_label: str = "95%"
) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(
        actual.index, lower, upper, color=_BLUE, alpha=0.25, label=f"{level_label} interval"
    )
    ax.plot(actual.index, actual.values, color="black", lw=0.7, label="actual")
    ax.plot(forecast.index, forecast.values, color=_BLUE, lw=0.8, label=f"{name} forecast")
    ax.set(
        title=f"{name} - conformal prediction intervals ({level_label})",
        xlabel="Date",
        ylabel="Load (MW)",
    )
    ax.legend()
    _save(fig, path)
