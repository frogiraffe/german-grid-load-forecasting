import pandas as pd

from loadfc.evaluation.slices import temporal_error_slices


def test_temporal_error_slices_cover_months_and_day_types():
    index = pd.date_range("2026-01-30", periods=5, freq="D")
    frame = pd.DataFrame(
        {"actual": [100.0] * 5, "forecast": [90.0, 100.0, 110.0, 100.0, 90.0]},
        index=index,
    )
    rows = temporal_error_slices(frame, "model", "test")
    month_rows = [row for row in rows if row["slice_type"] == "month"]
    day_rows = [row for row in rows if row["slice_type"] == "day_type"]
    assert {row["slice"] for row in month_rows} == {"2026-01", "2026-02"}
    assert {row["slice"] for row in day_rows} == {"weekday", "weekend"}
    assert sum(row["n"] for row in month_rows) == len(frame)
