"""Canonical UTC indexing and completeness checks for hourly inputs."""

from __future__ import annotations

import pandas as pd


def canonical_utc_index(
    frame: pd.Series | pd.DataFrame,
    *,
    source_timezone: str | None = None,
) -> pd.Series | pd.DataFrame:
    """Return an hourly object on a unique, complete UTC instant index."""

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("hourly data requires a DatetimeIndex")
    result = frame.sort_index().copy()
    if result.empty:
        raise ValueError("hourly data must not be empty")
    index = result.index
    if index.tz is None:
        if source_timezone is None:
            raise ValueError("naive hourly timestamps require a source timezone")
        try:
            index = index.tz_localize(source_timezone, ambiguous="infer", nonexistent="raise")
        except ValueError as error:
            raise ValueError(
                f"hourly timestamps are invalid or ambiguous in {source_timezone}"
            ) from error
    index = index.tz_convert("UTC")
    if index.has_duplicates:
        raise ValueError("hourly data contains duplicate UTC instants")
    off_grid = index[
        (index.minute != 0)
        | (index.second != 0)
        | (index.microsecond != 0)
        | (index.nanosecond != 0)
    ]
    if not off_grid.empty:
        sample = ", ".join(str(stamp) for stamp in off_grid[:3])
        raise ValueError(
            f"hourly data contains {len(off_grid)} off-grid timestamps (first: {sample})"
        )
    result.index = index
    expected = pd.date_range(index.min(), index.max(), freq="h", tz="UTC")
    missing = expected.difference(index)
    if not missing.empty:
        sample = ", ".join(str(stamp) for stamp in missing[:3])
        raise ValueError(f"hourly data is missing {len(missing)} UTC instants (first: {sample})")
    unexpected = index.difference(expected)
    if not unexpected.empty or len(index) != len(expected):
        sample = ", ".join(str(stamp) for stamp in unexpected[:3])
        raise ValueError(
            f"hourly data contains {len(unexpected)} unexpected UTC instants (first: {sample})"
        )
    result.index.name = "timestamp_utc"
    return result
