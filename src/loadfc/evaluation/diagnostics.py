"""Residual diagnostics for the SARIMAX model."""

from __future__ import annotations

import numpy as np
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch


def ljung_box(resid, lags: int = 10) -> tuple[float, float]:
    out = acorr_ljungbox(np.asarray(resid, dtype="float64"), lags=[lags], return_df=True)
    return float(out["lb_stat"].iloc[0]), float(out["lb_pvalue"].iloc[0])


def arch_lm(resid, lags: int = 7) -> tuple[float, float]:
    stat, p, _, _ = het_arch(np.asarray(resid, dtype="float64"), nlags=lags)
    return float(stat), float(p)
