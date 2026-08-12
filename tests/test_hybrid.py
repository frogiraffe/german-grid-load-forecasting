import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeRegressor

from loadfc.models.hybrid import HybridResidualRegressor


def test_hybrid_prediction_is_baseline_plus_learned_residual():
    X = np.arange(8, dtype="float64").reshape(-1, 1)
    y = np.full(8, 13.0)
    model = HybridResidualRegressor(
        DummyRegressor(strategy="constant", constant=10.0),
        DecisionTreeRegressor(random_state=42),
    ).fit(X, y)

    baseline, residual = model.predict_components(X)

    assert np.all(baseline == 10.0)
    assert np.all(residual == 3.0)
    assert np.all(model.predict(X) == 13.0)


def test_hybrid_rejects_prediction_before_fit():
    model = HybridResidualRegressor(DummyRegressor(), DecisionTreeRegressor(random_state=42))
    with pytest.raises(NotFittedError):
        model.predict([[1.0]])
