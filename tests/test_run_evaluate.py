from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from loadfc.evaluation.protocol import protocol_fingerprint

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_evaluate.py"


def _module():
    spec = importlib.util.spec_from_file_location("run_evaluate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cfg():
    return SimpleNamespace(
        models={"model": {"depth": 1}},
        seed=42,
        dataset_start=date(2019, 1, 14),
        ensemble={"members": ["model"]},
        split=SimpleNamespace(
            train_end=date(2024, 1, 1),
            val_end=date(2024, 1, 2),
            calibration_start=date(2024, 1, 3),
            calibration_end=date(2024, 1, 3),
            test_start=date(2024, 1, 4),
            test_end=date(2024, 1, 4),
        ),
    )


def test_frozen_daily_stream_fits_once_and_splits_only_after_continuous_forecast(monkeypatch):
    module = _module()
    cfg = _cfg()
    index = pd.Index(pd.date_range("2024-01-01", periods=4, freq="D").date)
    features = pd.DataFrame(
        {"daily_load": [100.0, 101.0, 102.0, 103.0], "feature": [1.0, 2.0, 3.0, 4.0]},
        index=index,
    )
    calls = []

    monkeypatch.setattr(module, "KINDS", {"model": "ml"})
    monkeypatch.setattr(module, "feature_matrix", lambda frame, family: frame)
    monkeypatch.setattr(module, "exog_columns", lambda family: ["feature"])
    monkeypatch.setattr(module, "_factory", lambda kind, config: lambda: object())
    monkeypatch.setattr(module, "baseline_predictions", lambda *args: {})

    def fake_rolling(model, train, evaluation, columns):
        calls.append((train.index.max(), evaluation.index.tolist(), columns))
        return (evaluation["daily_load"] - 1.0).rename("model")

    monkeypatch.setattr(module, "rolling_forecast", fake_rolling)

    predictions = module._frozen_daily_predictions(cfg, features)
    periods = module._split_frozen_predictions(predictions, cfg)

    assert calls == [(date(2024, 1, 2), [date(2024, 1, 3), date(2024, 1, 4)], ["feature"])]
    assert periods["calibration"]["model"].index.tolist() == [date(2024, 1, 3)]
    assert periods["retrospective_final"]["model"].index.tolist() == [date(2024, 1, 4)]


def test_daily_protocol_records_reference_validation_evidence_only(monkeypatch):
    module = _module()
    cfg = _cfg()
    monkeypatch.setattr(module, "KINDS", {"model": "ml"})
    monkeypatch.setattr(module, "exog_columns", lambda family: ["feature"])

    records = module._daily_protocol_records(
        cfg,
        source_revision="source",
        config_sha256="config",
        validation_evidence=Path("results/metrics/validation_metrics.csv"),
    )

    assert records["daily/model"]["validation_selection_evidence"].endswith(
        "validation_metrics.csv"
    )
    assert "calibration" not in records["daily/model"]["validation_selection_evidence"]
    assert "test" not in records["daily/model"]["validation_selection_evidence"]


def test_post_origin_weather_mutation_cannot_change_daily_selection(monkeypatch, tmp_path):
    module = _module()
    cfg = _cfg()
    index = pd.Index(pd.date_range("2024-01-01", periods=4, freq="D").date)
    baseline = pd.DataFrame(
        {
            "daily_load": [100.0, 101.0, 102.0, 103.0],
            "origin_available": [1.0, 2.0, 3.0, 4.0],
            "post_origin_weather": [10.0, 11.0, 12.0, 13.0],
        },
        index=index,
    )
    mutated = baseline.copy()
    mutated["post_origin_weather"] += 10_000.0

    monkeypatch.setattr(module, "KINDS", {"model": "ml"})
    monkeypatch.setattr(module, "feature_matrix", lambda frame, family: frame)
    monkeypatch.setattr(module, "exog_columns", lambda family: ["origin_available"])
    monkeypatch.setattr(module, "_factory", lambda kind, config: lambda: object())
    monkeypatch.setattr(module, "baseline_predictions", lambda *args: {})
    monkeypatch.setattr(
        module,
        "rolling_forecast",
        lambda model, train, evaluation, columns: pd.Series(
            evaluation[columns].sum(axis=1).to_numpy() + 100.0,
            index=evaluation.index,
            name="model",
        ),
    )

    first = module._frozen_daily_predictions(cfg, baseline)
    second = module._frozen_daily_predictions(cfg, mutated)
    records = module._daily_protocol_records(
        cfg,
        source_revision="source",
        config_sha256="config",
        validation_evidence=tmp_path / "validation.csv",
    )

    pd.testing.assert_series_equal(first["model"], second["model"])
    assert protocol_fingerprint(records["daily/model"]) == protocol_fingerprint(
        module._daily_protocol_records(
            cfg,
            source_revision="source",
            config_sha256="config",
            validation_evidence=tmp_path / "validation.csv",
        )["daily/model"]
    )
