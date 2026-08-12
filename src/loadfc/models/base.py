"""Common forecaster interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseForecaster(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, exog_cols: list[str]) -> None: ...

    @abstractmethod
    def predict_next(self, exog_row: pd.Series) -> float: ...

    def update(self, y_actual: float, exog_row: pd.Series) -> None:
        """Consume one realised outcome; default is stateless."""
        return None
