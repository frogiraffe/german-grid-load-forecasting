"""Variance inflation factors for the exogenous design."""

from __future__ import annotations

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


def compute_vif(df: pd.DataFrame) -> pd.Series:
    x = add_constant(df, has_constant="add")
    vifs = {
        col: variance_inflation_factor(x.values, i)
        for i, col in enumerate(x.columns)
        if col != "const"
    }
    return pd.Series(vifs).sort_values(ascending=False)
