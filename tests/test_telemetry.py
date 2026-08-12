from __future__ import annotations

import pandas as pd
import pytest

from loadfc.evaluation.telemetry import rolling_backtest_telemetry


def test_telemetry_raises_mape_and_coverage_alerts_after_full_window():
    predictions = pd.DataFrame(
        {
            "actual": [100.0] * 4,
            "forecast": [90.0] * 4,
            "lower": [95.0] * 4,
            "upper": [105.0] * 4,
        }
    )

    telemetry = rolling_backtest_telemetry(
        predictions,
        window=3,
        mape_warning=5.0,
        coverage_floor=0.85,
        lower_column="lower",
        upper_column="upper",
    )

    assert telemetry["mape_alert"].tolist() == [False, False, True, True]
    assert telemetry["coverage_alert"].tolist() == [False, False, False, False]
    assert telemetry["rolling_coverage"].iloc[-1] == 1.0


def test_telemetry_flags_undercoverage():
    predictions = pd.DataFrame(
        {
            "actual": [100.0] * 3,
            "forecast": [100.0] * 3,
            "lower": [101.0] * 3,
            "upper": [110.0] * 3,
        }
    )

    telemetry = rolling_backtest_telemetry(
        predictions,
        window=2,
        mape_warning=5.0,
        coverage_floor=0.85,
        lower_column="lower",
        upper_column="upper",
    )

    assert telemetry["coverage_alert"].tolist() == [False, True, True]


def test_telemetry_rejects_zero_actuals():
    with pytest.raises(ValueError, match="zero actuals"):
        rolling_backtest_telemetry(
            pd.DataFrame({"actual": [0.0], "forecast": [1.0]}),
            window=2,
            mape_warning=5.0,
            coverage_floor=0.85,
        )
