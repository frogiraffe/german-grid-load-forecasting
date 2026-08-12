from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

from loadfc.models.reconciled import ReconciledForecaster
from loadfc.tracking import sha256_file, track_sklearn_run


def test_track_sklearn_run_registers_loadable_candidate(tmp_path):
    inputs = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    estimator = LinearRegression().fit(inputs, np.array([1.0, 3.0, 5.0]))
    artifact = tmp_path / "metrics.csv"
    artifact.write_text("metric,value\nmae,1.0\n")

    tracked = track_sklearn_run(
        root=tmp_path,
        experiment_name="test-experiment",
        run_name="test-run",
        model_name="test-hourly-model",
        estimator=estimator,
        input_example=inputs,
        params={"seed": 42},
        metrics={"mae": 1.0},
        tags={"data_sha256": sha256_file(artifact)},
        artifact_paths=[artifact],
    )

    assert (tmp_path / "mlflow.db").exists()
    assert tracked.model_version == "1"

    import mlflow.sklearn

    mlflow.set_tracking_uri(tracked.tracking_uri)
    loaded = mlflow.sklearn.load_model("models:/test-hourly-model@candidate")
    np.testing.assert_allclose(loaded.predict(inputs), estimator.predict(inputs))


def test_registered_reconciled_model_reproduces_reconciled_predictions(tmp_path):
    hourly_input = pd.DataFrame({"hourly_feature": np.arange(24, dtype="float64")})
    hourly = LinearRegression().fit(
        hourly_input,
        100.0 + hourly_input["hourly_feature"].to_numpy(),
    )
    daily = DummyRegressor(strategy="constant", constant=240.0).fit(
        np.array([[0.0], [1.0]]),
        np.array([240.0, 240.0]),
    )
    reconciled = ReconciledForecaster(
        hourly_estimator=hourly,
        daily_estimator=daily,
        hourly_feature_columns=("hourly_feature",),
        daily_feature_columns=("daily_feature",),
    )
    inputs = hourly_input.copy()
    inputs["reconciliation_date"] = "2026-07-01"
    inputs["daily__daily_feature"] = 1.0
    expected = reconciled.predict(inputs)
    artifact = tmp_path / "metrics.csv"
    artifact.write_text("metric,value\nmae,1.0\n")

    tracked = track_sklearn_run(
        root=tmp_path,
        experiment_name="test-reconciled-experiment",
        run_name="test-reconciled-run",
        model_name="test-reconciled-model",
        estimator=reconciled,
        input_example=inputs,
        params={"seed": 42},
        metrics={"mae": 1.0},
        tags={"data_sha256": sha256_file(artifact)},
        artifact_paths=[artifact],
    )

    import mlflow.sklearn

    mlflow.set_tracking_uri(tracked.tracking_uri)
    loaded = mlflow.sklearn.load_model("models:/test-reconciled-model@candidate")
    np.testing.assert_allclose(loaded.predict(inputs), expected)
