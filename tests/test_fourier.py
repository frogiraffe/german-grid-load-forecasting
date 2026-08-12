from datetime import date, timedelta

import numpy as np
import pandas as pd

from loadfc.features.fourier import fourier_terms


def test_columns_named_by_label_and_harmonic():
    idx = pd.Index([date(2024, 1, 1) + timedelta(days=i) for i in range(10)])
    df = fourier_terms(idx, period=7, label="week", harmonics=1)
    assert list(df.columns) == ["sin_1_week", "cos_1_week"]


def test_weekly_terms_repeat_every_seven_days():
    idx = pd.Index([date(2024, 1, 1) + timedelta(days=i) for i in range(14)])
    df = fourier_terms(idx, period=7, label="week", harmonics=1)
    assert np.allclose(df.iloc[0].values, df.iloc[7].values)


def test_terms_within_unit_range():
    idx = pd.Index([date(2024, 1, 1) + timedelta(days=i) for i in range(400)])
    df = fourier_terms(idx, period=365.25, label="year", harmonics=1)
    assert df.to_numpy().min() >= -1.0 and df.to_numpy().max() <= 1.0
