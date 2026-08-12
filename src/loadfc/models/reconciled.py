"""Deployable reconciliation wrapper for hourly and daily fitted estimators."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class ReconciledForecaster(RegressorMixin, BaseEstimator):
    """Scale complete local-day hourly profiles to fitted daily-model anchors."""

    def __init__(
        self,
        hourly_estimator: Any,
        daily_estimator: Any,
        hourly_feature_columns: tuple[str, ...],
        daily_feature_columns: tuple[str, ...],
        *,
        date_column: str = "reconciliation_date",
        daily_prefix: str = "daily__",
        timezone: str = "Europe/Berlin",
    ) -> None:
        self.hourly_estimator = hourly_estimator
        self.daily_estimator = daily_estimator
        self.hourly_feature_columns = hourly_feature_columns
        self.daily_feature_columns = daily_feature_columns
        self.date_column = date_column
        self.daily_prefix = daily_prefix
        self.timezone = timezone

    def fit(self, X, y=None):
        """Keep the already-fitted component models unchanged."""

        self._validate_input(X)
        return self

    def predict(self, X) -> np.ndarray:
        """Return hourly predictions reconciled to one daily anchor per local day."""

        frame = self._validate_input(X)
        dates = pd.to_datetime(frame[self.date_column], errors="raise").dt.date
        raw = np.asarray(
            self.hourly_estimator.predict(frame[list(self.hourly_feature_columns)]),
            dtype="float64",
        )
        if raw.shape != (len(frame),) or not np.isfinite(raw).all():
            raise ValueError("hourly estimator must return one finite prediction per row")

        daily_columns = [f"{self.daily_prefix}{column}" for column in self.daily_feature_columns]
        daily_frame = frame[daily_columns].copy()
        daily_frame.columns = list(self.daily_feature_columns)
        daily_frame[self.date_column] = dates.to_numpy()
        grouped = daily_frame.groupby(self.date_column, sort=False, dropna=False)
        if (grouped[list(self.daily_feature_columns)].nunique(dropna=False) > 1).any().any():
            raise ValueError("daily anchor features must be constant within each local day")
        unique_daily = grouped[list(self.daily_feature_columns)].first()
        anchors = np.asarray(
            self.daily_estimator.predict(unique_daily.astype("float64")),
            dtype="float64",
        )
        if anchors.shape != (len(unique_daily),) or not np.isfinite(anchors).all():
            raise ValueError("daily estimator must return one finite anchor per local day")
        anchor_series = pd.Series(anchors, index=unique_daily.index)
        if (anchor_series <= 0).any():
            raise ValueError("daily anchors must be positive")

        local_dates = pd.Index(dates, name=self.date_column)
        counts = pd.Series(1, index=local_dates).groupby(level=0).sum()
        expected = pd.Series(
            {
                day: int(
                    (
                        pd.Timestamp(day + timedelta(days=1), tz=self.timezone)
                        - pd.Timestamp(day, tz=self.timezone)
                    )
                    / pd.Timedelta(hours=1)
                )
                for day in counts.index
            }
        )
        incomplete = counts[counts != expected]
        if not incomplete.empty:
            raise ValueError(
                f"reconciliation requires complete local days: {list(incomplete.index)}"
            )

        profile_means = pd.Series(raw, index=local_dates).groupby(level=0).mean()
        if not np.isfinite(profile_means.to_numpy()).all() or (profile_means <= 0).any():
            raise ValueError("hourly profile means must be finite and positive")
        factors = anchor_series.loc[profile_means.index] / profile_means
        return raw * local_dates.map(factors).to_numpy(dtype="float64")

    def _validate_input(self, X) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("reconciled forecasting requires a pandas DataFrame")
        daily_columns = {f"{self.daily_prefix}{column}" for column in self.daily_feature_columns}
        required = {
            *self.hourly_feature_columns,
            *daily_columns,
            self.date_column,
        }
        missing = required - set(X.columns)
        if missing:
            raise ValueError(f"reconciliation input is missing columns: {sorted(missing)}")
        if X[list(required)].isna().any().any():
            raise ValueError("reconciliation input contains missing values")
        return X
