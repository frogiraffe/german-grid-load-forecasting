from datetime import date

import pandas as pd

from loadfc.features.weather_features import add_degree_days, fit_temperature_climatology


def test_hdd_and_cdd_thresholds():
    df = pd.DataFrame({"Temp_t": [10.0, 18.0, 25.0]})
    out = add_degree_days(df, hdd_threshold=18.0, cdd_threshold=22.0)
    assert list(out["HDD"]) == [8.0, 0.0, 0.0]
    assert list(out["CDD"]) == [0.0, 0.0, 3.0]


def test_add_degree_days_does_not_mutate_input():
    df = pd.DataFrame({"Temp_t": [5.0]})
    add_degree_days(df, 18.0, 22.0)
    assert "HDD" not in df.columns


def test_climatology_is_frozen_before_validation_and_keeps_leap_axis():
    index = pd.Index([date(2020, 2, 28), date(2020, 2, 29), date(2020, 3, 1)])
    baseline = pd.Series([1.0, 2.0, 3.0], index=index)
    changed = baseline.copy()
    changed.loc[date(2020, 2, 29)] = 999.0

    curve = fit_temperature_climatology(baseline, date(2020, 2, 29))
    mutated_curve = fit_temperature_climatology(changed, date(2020, 2, 29))

    pd.testing.assert_series_equal(curve, mutated_curve)
    assert len(curve) == 366
    assert curve.loc[60] == curve.loc[60]
    assert curve.loc[200] == 1.0
