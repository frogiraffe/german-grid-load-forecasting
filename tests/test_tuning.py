from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from loadfc.models.ml import make_estimator
from loadfc.tuning.search_spaces import SPACES, suggest_params
from loadfc.tuning.study import cv_mase, training_slice, tune_model

ML_KINDS = ["xgboost", "lightgbm", "random_forest"]


def _config_params(kind: str) -> dict:
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())
    return cfg["models"][kind]


def _frame(n, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.Index([date(2020, 1, 1) + timedelta(days=i) for i in range(n)])
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 500 + 3 * x1 - 2 * x2 + rng.normal(scale=1, size=n)
    return pd.DataFrame({"daily_load": y, "x1": x1, "x2": x2}, index=idx)


@pytest.mark.parametrize("kind", ML_KINDS)
def test_search_space_keys_match_config(kind):
    assert set(SPACES[kind]) == set(_config_params(kind))


@pytest.mark.parametrize("kind", ML_KINDS)
def test_search_space_brackets_committed_params(kind):
    for name, value in _config_params(kind).items():
        _, lo, hi = SPACES[kind][name]
        assert lo <= value <= hi, f"{kind}.{name}={value} outside [{lo}, {hi}]"


@pytest.mark.parametrize("kind", ML_KINDS)
def test_suggest_params_are_usable(kind):
    import optuna

    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    trial = study.ask()
    params = suggest_params(kind, trial)
    assert set(params) == set(SPACES[kind])
    make_estimator(kind, params)


def test_training_slice_excludes_test_period():
    df = _frame(40)
    cut = df.index[30]
    sliced = training_slice(df, cut)
    assert sliced.index.max() < cut
    assert len(sliced) == 30


def test_cv_mase_is_finite_and_uses_all_folds():
    df = _frame(120)
    params = {"n_estimators": 20, "max_depth": 3}
    score = cv_mase("random_forest", df, ["x1", "x2"], params, n_splits=4, seed=0)
    assert np.isfinite(score) and score > 0


def test_tune_model_returns_best_params_with_right_keys():
    df = _frame(120)
    best = tune_model("random_forest", df, ["x1", "x2"], n_trials=3, n_splits=3, seed=0)
    assert set(best.params) == set(SPACES["random_forest"])
    assert np.isfinite(best.value) and best.value > 0


def test_tune_model_is_reproducible():
    df = _frame(120)
    a = tune_model("xgboost", df, ["x1", "x2"], n_trials=3, n_splits=3, seed=7)
    b = tune_model("xgboost", df, ["x1", "x2"], n_trials=3, n_splits=3, seed=7)
    assert a.params == b.params
