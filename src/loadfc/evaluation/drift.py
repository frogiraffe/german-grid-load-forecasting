"""Batch drift telemetry for features and sequential forecast residuals."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

_EPSILON = 1e-6


def population_stability_index(
    reference: Sequence[float],
    current: Sequence[float],
    *,
    bins: int = 10,
) -> float:
    """Measure numeric distribution shift using reference quantile bins."""

    if bins < 2:
        raise ValueError("PSI requires at least two bins")
    ref = np.asarray(reference, dtype="float64")
    cur = np.asarray(current, dtype="float64")
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if ref.size == 0 or cur.size == 0:
        raise ValueError("PSI requires finite reference and current observations")

    inner = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)[1:-1]))
    if inner.size == 1 and np.all(ref == inner[0]):
        margin = max(abs(float(inner[0])) * _EPSILON, _EPSILON)
        inner = np.array([inner[0] - margin, inner[0] + margin])
    edges = np.concatenate(([-np.inf], inner, [np.inf]))
    ref_counts = np.histogram(ref, bins=edges)[0] / ref.size
    cur_counts = np.histogram(cur, bins=edges)[0] / cur.size
    ref_share = np.clip(ref_counts, _EPSILON, None)
    cur_share = np.clip(cur_counts, _EPSILON, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def feature_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: Sequence[str],
    *,
    warning_threshold: float = 0.1,
    critical_threshold: float = 0.25,
) -> pd.DataFrame:
    """Return PSI and an operating status for each requested feature."""

    if not 0 <= warning_threshold < critical_threshold:
        raise ValueError("drift thresholds must satisfy 0 <= warning < critical")
    missing = set(columns) - (set(reference) & set(current))
    if missing:
        raise ValueError(f"drift columns are missing: {sorted(missing)}")
    rows = []
    for column in columns:
        psi = population_stability_index(reference[column], current[column])
        status = (
            "critical"
            if psi >= critical_threshold
            else "warning"
            if psi >= warning_threshold
            else "stable"
        )
        rows.append({"feature": column, "psi": psi, "status": status})
    return pd.DataFrame(rows).set_index("feature")


def page_hinkley(
    values: Sequence[float],
    *,
    delta: float = 0.0,
    threshold: float = 50.0,
    reference_mean: float | None = None,
) -> pd.DataFrame:
    """Track upward mean shifts without using future residuals."""

    if delta < 0 or threshold <= 0:
        raise ValueError("Page-Hinkley requires delta >= 0 and threshold > 0")
    observations = np.asarray(values, dtype="float64")
    if observations.size == 0 or not np.isfinite(observations).all():
        raise ValueError("Page-Hinkley requires finite observations")

    running_mean = 0.0
    cumulative = 0.0
    minimum = 0.0
    rows = []
    for step, value in enumerate(observations, start=1):
        running_mean += (float(value) - running_mean) / step
        center = running_mean if reference_mean is None else float(reference_mean)
        cumulative += float(value) - center - delta
        minimum = min(minimum, cumulative)
        statistic = cumulative - minimum
        rows.append(
            {
                "value": float(value),
                "running_mean": running_mean,
                "statistic": statistic,
                "alert": statistic > threshold,
            }
        )
    return pd.DataFrame(rows)
