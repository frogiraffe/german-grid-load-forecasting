import warnings

import numpy as np
import pandas as pd
import pytest

from loadfc.viz import plots

with warnings.catch_warnings():
    warnings.simplefilter("ignore", PendingDeprecationWarning)
    from scripts import run_shap


def test_residual_plot_limits_acf_lags_to_short_series(tmp_path):
    residuals = pd.Series(
        np.linspace(-2.0, 2.0, 10),
        index=pd.date_range("2026-07-01", periods=10),
    )
    destination = tmp_path / "residuals.png"
    plots.sarimax_residuals(residuals, destination, acf_lags=30)
    assert destination.exists()
    assert destination.stat().st_size > 0


def test_residual_plot_requires_two_observations(tmp_path):
    with pytest.raises(ValueError, match="two"):
        plots.sarimax_residuals(pd.Series([1.0]), tmp_path / "unused.png")


def test_timeseries_labels_final_boundary_as_retrospective(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(plots, "_save", lambda fig, _: captured.update(fig=fig))
    load = pd.Series(
        [1.0, 2.0],
        index=pd.date_range("2025-12-31", periods=2),
    )

    plots.timeseries(load, pd.Timestamp("2026-01-01").date(), tmp_path / "unused.png")

    labels = captured["fig"].axes[0].get_legend_handles_labels()[1]
    assert labels == ["retrospective final starts"]


def test_forecast_title_labels_inspected_period_as_retrospective(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(plots, "_save", lambda fig, _: captured.update(fig=fig))
    actual = pd.Series([1.0, 2.0], index=pd.date_range("2026-01-01", periods=2))

    plots.forecast_vs_actual(actual, actual, "Ensemble", tmp_path / "unused.png")

    assert captured["fig"].axes[0].get_title().startswith(
        "Ensemble - retrospective final forecast vs actual"
    )


def test_model_comparison_title_labels_metrics_as_retrospective(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(plots, "_save", lambda fig, _: captured.update(fig=fig))
    metrics = pd.DataFrame({"MAE": [1.0, 2.0]}, index=["a", "b"])

    plots.model_comparison(metrics, tmp_path / "unused.png")

    assert captured["fig"].axes[0].get_title() == "Model comparison - retrospective final metrics"


def test_shap_beeswarm_title_labels_period_as_retrospective(monkeypatch, tmp_path):
    titles = []
    monkeypatch.setattr(run_shap.shap.plots, "beeswarm", lambda *_, **__: None)
    monkeypatch.setattr(run_shap.plt, "title", titles.append)
    monkeypatch.setattr(run_shap.plt, "savefig", lambda *_, **__: None)

    run_shap._beeswarm(object(), tmp_path / "unused.png")

    assert titles == ["Per-observation SHAP effects — XGBoost (retrospective final period)"]
