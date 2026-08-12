"""Long-format supervised rows for direct global multi-horizon models."""

from __future__ import annotations

import pandas as pd


def direct_horizon_frame(hourly_features: pd.DataFrame, *, horizon: int = 24) -> pd.DataFrame:
    """Repeat valid-time features for every eligible day-ahead forecast origin."""

    if horizon < 1:
        raise ValueError("horizon must be positive")
    if "hourly_load" not in hourly_features:
        raise ValueError("hourly features require an hourly_load target")
    if not isinstance(hourly_features.index, pd.DatetimeIndex):
        raise ValueError("hourly features require a DatetimeIndex")

    first = hourly_features.index.min()
    blocks: list[pd.DataFrame] = []
    for step in range(1, horizon + 1):
        block = hourly_features.copy()
        block["horizon"] = step
        block["forecast_origin"] = block.index - pd.Timedelta(hours=step)
        block = block[block["forecast_origin"] >= first]
        block["valid_time"] = block.index
        blocks.append(block)
    out = pd.concat(blocks, ignore_index=True)
    return out.set_index(["forecast_origin", "valid_time", "horizon"], drop=False).sort_index()
