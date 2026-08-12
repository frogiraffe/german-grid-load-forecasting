"""Temporal forecast-error slices used by release diagnostics."""

from __future__ import annotations

import pandas as pd

from .metrics import mae, mape


def temporal_error_slices(frame: pd.DataFrame, model: str, period: str) -> list[dict]:
    evaluated = frame.copy()
    evaluated["error"] = evaluated["forecast"] - evaluated["actual"]
    evaluated["month"] = evaluated.index.strftime("%Y-%m")
    evaluated["day_type"] = [
        "weekend" if day.weekday() >= 5 else "weekday" for day in evaluated.index
    ]
    rows: list[dict] = []
    for slice_type in ["month", "day_type"]:
        for label, group in evaluated.groupby(slice_type):
            rows.append(
                {
                    "period": period,
                    "model": model,
                    "slice_type": slice_type,
                    "slice": label,
                    "n": len(group),
                    "MAE": mae(group["actual"], group["forecast"]),
                    "MAPE": mape(group["actual"], group["forecast"]),
                    "bias_MW": float(group["error"].mean()),
                }
            )
    return rows
