from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from loadfc.evaluation.hourly import (
    aggregate_horizon_to_daily,
    metrics_by_horizon,
    reconcile_to_daily_means,
    reconciliation_invariant_rows,
)


def _h24_predictions() -> pd.DataFrame:
    valid_time = pd.date_range("2024-01-02", periods=24, freq="h", tz="Europe/Berlin")
    index = pd.MultiIndex.from_arrays(
        [
            valid_time.tz_convert("UTC") - pd.Timedelta(hours=24),
            valid_time.tz_convert("UTC"),
            np.full(24, 24),
        ],
        names=["forecast_origin", "valid_time", "horizon"],
    )
    return pd.DataFrame(
        {
            "prediction": 80.0 + np.arange(24),
            "actual": 90.0 + np.arange(24),
        },
        index=index,
    )


def _partial_day_predictions() -> pd.DataFrame:
    valid_time = pd.date_range("2024-01-02", periods=3, freq="h", tz="Europe/Berlin")
    index = pd.MultiIndex.from_arrays(
        [
            valid_time.tz_convert("UTC") - pd.Timedelta(hours=24),
            valid_time.tz_convert("UTC"),
            np.full(3, 24),
        ],
        names=["forecast_origin", "valid_time", "horizon"],
    )
    return pd.DataFrame(
        {"prediction": [80.0, 81.0, 82.0], "actual": [90.0, 91.0, 92.0]},
        index=index,
    )


def test_metrics_are_reported_for_each_horizon():
    metrics = metrics_by_horizon(_h24_predictions().assign(error=np.arange(24)))
    assert list(metrics.index) == [24]
    assert (metrics["n"] == 24).all()
    assert (metrics[["mae", "rmse"]] >= 0).all().all()


def test_metrics_require_evaluated_errors():
    with pytest.raises(ValueError, match="error column"):
        metrics_by_horizon(pd.DataFrame({"prediction": [1.0]}))


def test_fixed_horizon_aggregates_by_german_local_day():
    daily = aggregate_horizon_to_daily(_h24_predictions(), horizon=24)
    assert len(daily) == 1
    assert list(daily) == ["actual", "prediction"]


def test_reconciliation_preserves_profile_and_matches_daily_anchor():
    predictions = _h24_predictions()
    date = pd.Timestamp("2024-01-02")
    original = predictions["prediction"]
    anchor = pd.Series([200.0], index=[date])

    reconciled = reconcile_to_daily_means(predictions, anchor, horizon=24)

    assert reconciled["prediction"].mean() == pytest.approx(200.0)
    assert reconciled["daily_anchor"].eq(200.0).all()
    assert (reconciled["prediction"] / reconciled["prediction_unreconciled"]).nunique() == 1
    assert reconciled["prediction_unreconciled"].to_numpy() == pytest.approx(original.to_numpy())
    assert reconciled["error"].to_numpy() == pytest.approx(
        reconciled["actual"] - reconciled["prediction"]
    )


def test_reconciliation_rejects_missing_daily_anchor():
    with pytest.raises(ValueError, match="missing anchors"):
        reconcile_to_daily_means(
            _h24_predictions(),
            pd.Series([200.0], index=[pd.Timestamp("2024-01-03")]),
            horizon=24,
        )


def test_reconciliation_rejects_nonfinite_hourly_prediction():
    predictions = _h24_predictions()
    predictions.iloc[0, predictions.columns.get_loc("prediction")] = np.nan

    with pytest.raises(ValueError, match="hourly predictions must be finite"):
        reconcile_to_daily_means(
            predictions,
            pd.Series([200.0], index=[pd.Timestamp("2024-01-02")]),
            horizon=24,
        )


def test_reconciliation_rejects_incomplete_local_day():
    with pytest.raises(ValueError, match="incomplete local days"):
        reconcile_to_daily_means(
            _partial_day_predictions(),
            pd.Series([200.0], index=[pd.Timestamp("2024-01-02")]),
            horizon=24,
        )


@pytest.mark.parametrize(
    ("local_day", "expected_hours"),
    [("2024-03-31", 23), ("2024-01-02", 24), ("2024-10-27", 25)],
)
def test_reconciliation_accepts_complete_berlin_days(local_day, expected_hours):
    start = pd.Timestamp(local_day, tz="Europe/Berlin")
    end = pd.Timestamp(start.date() + timedelta(days=1), tz="Europe/Berlin")
    valid_time = pd.date_range(start, end, freq="h", inclusive="left")
    utc = valid_time.tz_convert("UTC")
    index = pd.MultiIndex.from_arrays(
        [utc - pd.Timedelta(hours=24), utc, np.full(len(utc), 24)],
        names=["forecast_origin", "valid_time", "horizon"],
    )
    predictions = pd.DataFrame(
        {"actual": np.arange(len(utc), dtype="float64"), "prediction": np.arange(len(utc)) + 1.0},
        index=index,
    )
    anchor = pd.Series([200.0], index=[start.date()])

    reconciled = reconcile_to_daily_means(predictions, anchor, horizon=24)

    assert len(utc) == expected_hours
    assert utc.is_unique
    assert reconciled["prediction"].mean() == pytest.approx(200.0)
    if expected_hours == 25:
        assert (valid_time.hour == 2).sum() == 2


def test_reconciliation_rejects_duplicate_utc_identity():
    predictions = _h24_predictions()
    duplicate = pd.concat([predictions, predictions.iloc[[0]]])

    with pytest.raises(ValueError, match="unique valid times"):
        reconcile_to_daily_means(
            duplicate,
            pd.Series([200.0], index=[pd.Timestamp("2024-01-02")]),
            horizon=24,
        )


def test_reconciliation_invariant_rows_report_anchor_delta_and_tolerance():
    reconciled = reconcile_to_daily_means(
        _h24_predictions(), pd.Series([200.0], index=[pd.Timestamp("2024-01-02")])
    )
    rows = reconciliation_invariant_rows(reconciled, tolerance=1e-9)
    assert list(rows) == [
        "local_date",
        "reconciled_hourly_mean",
        "daily_anchor",
        "delta",
        "abs_delta",
        "tolerance",
        "pass",
        "n",
    ]
    assert rows.loc[0, "pass"]
    assert rows.loc[0, "n"] == 24
    assert rows.loc[0, "abs_delta"] <= rows.loc[0, "tolerance"]


def test_reconciliation_invariant_rows_fail_on_tolerance_violation():
    reconciled = reconcile_to_daily_means(
        _h24_predictions(), pd.Series([200.0], index=[pd.Timestamp("2024-01-02")])
    )
    reconciled.loc[reconciled.index[0], "prediction"] += 1.0
    rows = reconciliation_invariant_rows(reconciled, tolerance=1e-9)
    assert not rows.loc[0, "pass"]
    assert rows.loc[0, "abs_delta"] > rows.loc[0, "tolerance"]
