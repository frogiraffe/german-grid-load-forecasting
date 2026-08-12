import pandas as pd
import pytest

from loadfc.features.horizon import direct_horizon_frame


def test_direct_horizon_frame_has_24_ordered_forecast_steps():
    index = pd.date_range("2024-01-01", periods=30, freq="h", tz="UTC")
    features = pd.DataFrame({"hourly_load": range(30), "L_t-24": range(30)}, index=index)

    frame = direct_horizon_frame(features)

    assert set(frame.index.get_level_values("horizon")) == set(range(1, 25))
    assert "horizon" in frame.columns
    origins = frame.index.get_level_values("forecast_origin")
    valid_times = frame.index.get_level_values("valid_time")
    horizons = frame.index.get_level_values("horizon")
    assert all(
        valid == origin + pd.Timedelta(hours=int(step))
        for origin, valid, step in zip(origins, valid_times, horizons, strict=True)
    )


def test_direct_horizon_frame_rejects_invalid_horizon():
    index = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    frame = pd.DataFrame({"hourly_load": [1.0, 2.0]}, index=index)
    with pytest.raises(ValueError, match="positive"):
        direct_horizon_frame(frame, horizon=0)
