from datetime import date, timedelta

import pandas as pd

from loadfc.features.lags import add_lags


def test_lags_shift_and_name():
    idx = [date(2024, 1, 1) + timedelta(days=i) for i in range(9)]
    s = pd.Series(range(100, 109), index=idx, dtype="float64", name="daily_load")
    out = add_lags(s, [1, 7])
    assert list(out.columns) == ["L_t-1", "L_t-7"]
    assert out["L_t-1"].iloc[1] == 100.0
    assert out["L_t-7"].iloc[7] == 100.0
    assert pd.isna(out["L_t-1"].iloc[0])
    assert out["L_t-7"].iloc[:7].isna().all()
