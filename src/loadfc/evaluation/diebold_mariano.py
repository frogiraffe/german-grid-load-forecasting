"""Diebold-Mariano test of equal predictive accuracy (absolute-error loss)."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def _hac_variance(d: np.ndarray, bandwidth: int) -> float:
    d = d - d.mean()
    n = len(d)
    total = np.dot(d, d) / n
    for lag in range(1, bandwidth + 1):
        weight = 1.0 - lag / (bandwidth + 1)
        cov = np.dot(d[lag:], d[:-lag]) / n
        total += 2.0 * weight * cov
    return total


def dm_test(errors1, errors2, bandwidth: int | None = None) -> tuple[float, float]:
    e1 = np.asarray(errors1, dtype="float64")
    e2 = np.asarray(errors2, dtype="float64")
    d = np.abs(e1) - np.abs(e2)
    n = len(d)
    if bandwidth is None:
        bandwidth = int(np.floor(n ** (1.0 / 3.0)))
    var = _hac_variance(d, bandwidth)
    if var <= 0:
        return 0.0, 1.0
    stat = d.mean() / np.sqrt(var / n)
    p = 2.0 * (1.0 - stats.norm.cdf(abs(stat)))
    return float(stat), float(p)


def dm_pvalue_matrix(errors: dict[str, np.ndarray], bonferroni: bool = True) -> pd.DataFrame:
    names = list(errors)
    n_pairs = len(list(combinations(names, 2)))
    m = pd.DataFrame(np.nan, index=names, columns=names)
    for a, b in combinations(names, 2):
        _, p = dm_test(errors[a], errors[b])
        if bonferroni:
            # Bonferroni correction across all model pairs, capped at 1.0.
            p = min(1.0, p * n_pairs)
        m.loc[a, b] = p
        m.loc[b, a] = p
    return m
