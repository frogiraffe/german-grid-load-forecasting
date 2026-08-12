from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_telemetry.py"


def _module():
    spec = importlib.util.spec_from_file_location("run_telemetry", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _predictions() -> pd.DataFrame:
    index = pd.date_range("2026-07-01", periods=3, freq="D", name="date")
    return pd.DataFrame({"actual": [100.0] * 3, "forecast": [99.0] * 3}, index=index)


def test_join_intervals_requires_identical_indexes():
    predictions = _predictions()
    shifted = predictions.index + pd.Timedelta(days=1)
    intervals = pd.DataFrame({"lower_95": [90.0] * 3, "upper_95": [110.0] * 3}, index=shifted)
    intervals.index.name = "date"

    with pytest.raises(AssertionError):
        _module()._join_intervals(predictions, intervals)


def test_join_intervals_rejects_nonfinite_bounds():
    predictions = _predictions()
    intervals = pd.DataFrame(
        {"lower_95": [90.0, np.nan, 90.0], "upper_95": [110.0] * 3},
        index=predictions.index,
    )

    with pytest.raises(ValueError, match="finite and complete"):
        _module()._join_intervals(predictions, intervals)
