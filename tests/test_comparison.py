from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from loadfc.evaluation.comparison import (
    assert_compatible_daily_artifacts,
    compare_daily_artifacts,
    compare_hourly_artifacts,
    paired_local_day_bootstrap,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_comparison.py"


def _module():
    spec = importlib.util.spec_from_file_location("run_comparison", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _artifact(name: str, dates=("2026-01-01", "2026-01-02"), offset=0.0):
    return pd.DataFrame(
        {
            "date": dates,
            "actual": [10.0, 20.0],
            "forecast": [11.0 + offset, 18.0 + offset],
            "evaluation_period": "retrospective_final",
            "stream_id": f"daily/{name}",
            "protocol_fingerprint": "abc123",
        }
    )


def test_daily_comparison_propagates_metadata_and_metrics():
    result = compare_daily_artifacts({"model": _artifact("model")}, naive_mae=2.0)
    row = result.iloc[0]
    assert row["stream_id"] == "daily/model"
    assert row["protocol_fingerprint"] == "abc123"
    assert row["dates"] == "2026-01-01:2026-01-02"
    assert row["n"] == 2
    assert row["MAE"] == pytest.approx(1.5)
    assert row["MASE"] == pytest.approx(0.75)


def test_daily_comparison_rejects_empty_or_mismatched_periods():
    with pytest.raises(ValueError, match="at least one"):
        compare_daily_artifacts({}, naive_mae=1.0)
    mismatched = _artifact("other")
    mismatched["evaluation_period"] = "calibration"
    with pytest.raises(ValueError, match="incompatible"):
        assert_compatible_daily_artifacts({"one": _artifact("one"), "two": mismatched})


def test_daily_comparison_rejects_non_overlapping_invalid_rows():
    frame = _artifact("bad", dates=("2026-01-01", "2026-01-01"))
    with pytest.raises(ValueError, match="valid and unique"):
        compare_daily_artifacts({"bad": frame}, naive_mae=1.0)


def _hourly(name: str, offset: float = 0.0) -> pd.DataFrame:
    valid = pd.date_range("2026-01-31 23:00", periods=48, freq="h", tz="UTC")
    origin = valid - pd.Timedelta(hours=24)
    return pd.DataFrame(
        {
            "forecast_origin": origin,
            "valid_time": valid,
            "horizon": 24,
            "actual": [10.0, 12.0, 11.0, 13.0] * 12,
            "forecast": [10.0 + offset, 11.0 + offset, 12.0 + offset, 12.0 + offset] * 12,
            "evaluation_period": "retrospective_final",
            "stream_id": f"hourly/comparison/{name}",
            "protocol_fingerprint": "abc123",
        }
    )


def test_hourly_comparison_uses_common_utc_rows_and_local_slices():
    candidate = _hourly("candidate")
    reference = _hourly("reference")
    reference = reference.iloc[:-1]
    result = compare_hourly_artifacts(
        {"candidate": candidate, "reference": reference}, naive_mae=2.0
    )
    aggregate = result[(result.model == "candidate") & (result.slice_type == "aggregate")].iloc[0]
    assert aggregate["n"] == 47
    assert set(result["slice_type"]) >= {"aggregate", "horizon", "local_hour", "local_day"}


def test_hourly_bootstrap_is_deterministic_and_marks_practical_ties():
    artifacts = {"reference": _hourly("reference"), "candidate": _hourly("candidate", 0.01)}
    first = paired_local_day_bootstrap(
        artifacts,
        reference="reference",
        selected_model="reference",
        seed=7,
        n_bootstrap=100,
        block_size=2,
    )
    second = paired_local_day_bootstrap(
        artifacts,
        reference="reference",
        selected_model="reference",
        seed=7,
        n_bootstrap=100,
        block_size=2,
    )
    pd.testing.assert_frame_equal(first, second)
    assert bool(first.iloc[0]["practical_tie"])
    assert bool(first.iloc[0]["selected_model_preserved"])


def test_hourly_comparison_persists_selection_metadata(tmp_path, monkeypatch):
    module = _module()
    results = tmp_path / "results"
    hourly = results / "hourly"
    (results / "metrics").mkdir(parents=True)
    hourly.mkdir()
    selection = pd.DataFrame(
        [
            {
                "selected_model": "residual_hybrid",
                "selection_metric": "reconciled_h24_hourly_MAE",
            }
        ]
    )
    selection.to_csv(hourly / "model_selection.csv", index=False)
    (results / "evaluation_protocol.json").write_text(
        '{"records":{"daily/daily":{},"hourly/point/residual_hybrid":{}}}'
    )
    cfg = SimpleNamespace(
        split=SimpleNamespace(test_start=pd.Timestamp("2026-01-01").date()),
        seed=42,
        path=lambda key: results if key == "results_dir" else tmp_path / key,
    )
    comparison = pd.DataFrame(
        [
            {
                "model": "residual_hybrid",
                "slice_type": "aggregate",
                "slice_value": "all",
                "n": 24,
            }
        ]
    )
    bootstrap = pd.DataFrame([{"candidate": "other", "reference": "residual_hybrid"}])
    monkeypatch.setattr(module.Config, "from_yaml", lambda path: cfg)
    monkeypatch.setattr(
        module,
        "_load_artifacts",
        lambda path: {
            "daily": pd.DataFrame(
                [{"stream_id": "daily/daily", "protocol_fingerprint": "fingerprint"}]
            )
        },
    )
    monkeypatch.setattr(module, "_load_hourly_artifacts", lambda path, records: {"residual_hybrid": pd.DataFrame()})
    monkeypatch.setattr(module, "assert_compatible_daily_artifacts", lambda artifacts: None)
    monkeypatch.setattr(module.pd, "read_parquet", lambda path: pd.DataFrame({"hourly_load": [1.0]}, index=pd.to_datetime(["2025-01-01"], utc=True)))
    monkeypatch.setattr(module, "build_features", lambda dataset, config: pd.DataFrame({"daily_load": [1.0]}, index=[pd.Timestamp("2025-01-01").date()]))
    monkeypatch.setattr(module, "seasonal_naive_mae", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(module, "protocol_fingerprint", lambda record: "fingerprint")
    monkeypatch.setattr(module, "compare_daily_artifacts", lambda *args, **kwargs: pd.DataFrame([{"model": "daily"}]))
    monkeypatch.setattr(module, "compare_hourly_artifacts", lambda *args, **kwargs: comparison.copy())
    monkeypatch.setattr(module, "paired_local_day_bootstrap", lambda *args, **kwargs: bootstrap.copy())
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--config", str(tmp_path / "config.yaml")])

    module.main()

    persisted = pd.read_csv(hourly / "hourly_comparison.csv")
    assert persisted[
        [
            "validation_selection_metric",
            "validation_selected_model",
            "validation_selection_evidence",
        ]
    ].iloc[0].to_dict() == {
        "validation_selection_metric": "reconciled_h24_hourly_MAE",
        "validation_selected_model": "residual_hybrid",
        "validation_selection_evidence": "results/hourly/model_ablation_validation_reconciled.csv",
    }
