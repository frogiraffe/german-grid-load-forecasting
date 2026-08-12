"""Trigonometric (Fourier) seasonality terms."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date

import numpy as np
import pandas as pd


def fourier_terms(
    index: Iterable[date], period: float, label: str, harmonics: int = 1
) -> pd.DataFrame:
    idx = pd.Index(index)
    t = np.array([d.toordinal() for d in idx], dtype="float64")
    out: dict[str, np.ndarray] = {}
    for k in range(1, harmonics + 1):
        out[f"sin_{k}_{label}"] = np.sin(2.0 * np.pi * k * t / period)
        out[f"cos_{k}_{label}"] = np.cos(2.0 * np.pi * k * t / period)
    return pd.DataFrame(out, index=idx)
