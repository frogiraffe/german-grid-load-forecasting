"""Baseline-plus-residual regression for extrapolation-aware load forecasts."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted


class HybridResidualRegressor(RegressorMixin, BaseEstimator):
    """Fit a baseline first, then train a second model on its residuals."""

    def __init__(self, baseline, residual_model):
        self.baseline = baseline
        self.residual_model = residual_model

    def fit(self, X, y):
        target = np.asarray(y, dtype="float64")
        self.baseline_ = clone(self.baseline).fit(X, target)
        baseline_prediction = np.asarray(self.baseline_.predict(X), dtype="float64")
        self.residual_model_ = clone(self.residual_model).fit(X, target - baseline_prediction)
        return self

    def predict_components(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Return baseline and learned residual contributions separately."""

        check_is_fitted(self, ("baseline_", "residual_model_"))
        baseline = np.asarray(self.baseline_.predict(X), dtype="float64")
        residual = np.asarray(self.residual_model_.predict(X), dtype="float64")
        return baseline, residual

    def predict(self, X) -> np.ndarray:
        baseline, residual = self.predict_components(X)
        return baseline + residual
