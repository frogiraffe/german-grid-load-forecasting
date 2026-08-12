from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_analysis.py"


def _module():
    spec = importlib.util.spec_from_file_location("run_analysis", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _features() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    frame = pd.DataFrame(
        {
            "daily_load": 50_000.0 + np.arange(len(index)),
            "Temp_forecast": np.resize(np.arange(-9.0, 35.0, 5.0), len(index)),
            "Temp_t": np.full(len(index), 999.0),
        },
        index=index,
    )
    return frame


def test_load_profiles_are_complete_ordered_descriptive_evidence():
    module = _module()
    features = _features()

    weekday = module.load_profile_weekday(features)
    month = module.load_profile_month(features)

    assert list(weekday) == [
        "weekday_order",
        "weekday",
        "mean_load_MW",
        "n_days",
        "evidence_scope",
    ]
    assert weekday["weekday"].tolist() == [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    assert weekday["weekday_order"].tolist() == list(range(1, 8))
    assert list(month) == [
        "month_order",
        "month",
        "mean_load_MW",
        "n_days",
        "evidence_scope",
    ]
    assert month["month_order"].tolist() == list(range(1, 13))
    assert month["month"].tolist() == [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    assert weekday["n_days"].sum() == len(features)
    assert month["n_days"].sum() == len(features)
    assert set(weekday["evidence_scope"]) == {"full_available_history_descriptive"}
    assert set(month["evidence_scope"]) == {"full_available_history_descriptive"}


def test_temperature_curve_uses_only_finite_forecast_origin_pairs():
    module = _module()
    features = _features().iloc[:11].copy()
    features.iloc[9, features.columns.get_loc("Temp_forecast")] = np.nan
    features.iloc[10, features.columns.get_loc("daily_load")] = np.inf

    curve = module.temperature_load_curve(features)

    assert list(curve) == [
        "bin_order",
        "lower_C",
        "upper_C",
        "mean_load_MW",
        "n_days",
        "evidence_scope",
    ]
    assert curve[["lower_C", "upper_C"]].to_records(index=False).tolist() == [
        (-10, -5),
        (-5, 0),
        (0, 5),
        (5, 10),
        (10, 15),
        (15, 20),
        (20, 25),
        (25, 30),
        (30, 35),
    ]
    assert curve["bin_order"].tolist() == list(range(1, 10))
    assert curve["n_days"].sum() == 9
    assert curve["n_days"].eq(1).all()
    assert set(curve["evidence_scope"]) == {"full_available_history_descriptive"}
    assert curve["mean_load_MW"].max() < 51_000


@pytest.mark.parametrize("temperature", [-10.01, 35.0])
def test_temperature_curve_rejects_eligible_values_outside_fixed_domain(temperature):
    features = _features().iloc[:9].copy()
    features.iloc[0, features.columns.get_loc("Temp_forecast")] = temperature

    with pytest.raises(ValueError, match="outside"):
        _module().temperature_load_curve(features)
