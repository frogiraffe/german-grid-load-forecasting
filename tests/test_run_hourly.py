from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from loadfc.config import Config
from loadfc.evaluation.conformal import horizon_cqr_intervals
from loadfc.tracking import sha256_file

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_hourly.py"
CONFIG = Path(__file__).parent / "fixtures" / "config_test.yaml"


def _module():
    spec = importlib.util.spec_from_file_location("run_hourly", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_date_mask_uses_german_calendar_boundaries():
    values = pd.Series(
        pd.to_datetime(["2023-12-31T22:00Z", "2023-12-31T23:00Z", "2024-01-01T23:00Z"])
    )
    day = pd.Timestamp("2024-01-01").date()

    mask = _module()._date_mask(values, day, day)

    assert mask.tolist() == [False, True, False]


def test_config_path_resolves_from_caller_without_duplicating_parent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    relative = Path("configs") / "production.yaml"

    resolved = Path(relative).resolve()

    assert resolved == tmp_path / relative


def test_reconciled_quantile_frame_rejects_crossing_after_reconciliation():
    valid_time = pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC")
    index = pd.MultiIndex.from_arrays(
        [valid_time - pd.Timedelta(hours=24), valid_time, [24, 24]],
        names=["forecast_origin", "valid_time", "horizon"],
    )
    point = pd.DataFrame(
        {
            "actual": [100.0, 110.0],
            "prediction": [102.0, 108.0],
            "horizon": [24, 24],
            "reconciliation_factor": [2.0, 0.5],
        },
        index=index,
    )
    lower = pd.DataFrame({"prediction": [90.0, 120.0]}, index=index)
    upper = pd.DataFrame({"prediction": [110.0, 100.0]}, index=index)

    with pytest.raises(ValueError, match="quantile predictions cross"):
        _module()._reconciled_quantile_frame(point, lower, upper)


def test_coverage_by_local_hour_reports_heterogeneous_width():
    valid_time = pd.to_datetime(["2024-01-01T02:00Z", "2024-01-01T18:00Z"])
    index = pd.MultiIndex.from_arrays(
        [valid_time - pd.Timedelta(hours=24), valid_time, [24, 24]],
        names=["forecast_origin", "valid_time", "horizon"],
    )
    intervals = pd.DataFrame(
        {
            "actual": [100.0, 100.0],
            "lower": [95.0, 80.0],
            "upper": [105.0, 120.0],
        },
        index=index,
    )

    report = _module()._coverage_by_local_hour(intervals)

    assert report.loc[3, "mean_width"] == 10.0
    assert report.loc[19, "mean_width"] == 40.0


def _prediction_frame(errors: list[float]) -> pd.DataFrame:
    valid_time = pd.date_range(
        "2024-01-01",
        periods=len(errors),
        freq="24h",
        tz="UTC",
    )
    index = pd.MultiIndex.from_arrays(
        [valid_time - pd.Timedelta(hours=24), valid_time, [24] * len(errors)],
        names=["forecast_origin", "valid_time", "horizon"],
    )
    actual = np.full(len(errors), 100.0)
    return pd.DataFrame(
        {
            "actual": actual,
            "prediction": actual - np.asarray(errors),
            "horizon": [24] * len(errors),
        },
        index=index,
    )


def test_temporal_model_ablation_reports_month_and_bias():
    report = _module()._temporal_model_ablation({"model": _prediction_frame([1.0, -3.0])})

    assert report.loc[("2024-01", "model"), "MAE"] == 2.0
    assert report.loc[("2024-01", "model"), "bias"] == -1.0


def test_interval_comparison_includes_adaptive_when_provided():
    module = _module()
    frame = pd.DataFrame({"actual": [1.0], "lower": [0.0], "upper": [2.0]})
    baseline = pd.DataFrame({"actual": [1.0], "lower": [5.0], "upper": [6.0]})

    without = module._interval_comparison(baseline, baseline)
    with_adaptive = module._interval_comparison(baseline, baseline, frame)

    assert "adaptive" not in without.index
    assert with_adaptive.index.tolist() == ["symmetric", "cqr", "adaptive"]
    assert with_adaptive.loc["adaptive", "coverage"] == 1.0
    assert with_adaptive.loc["adaptive", "mean_width"] == 2.0


def _interval_frame(valid_time: pd.DatetimeIndex) -> pd.DataFrame:
    horizons = np.resize(np.arange(1, 25), len(valid_time))
    index = pd.MultiIndex.from_arrays(
        [valid_time - pd.Timedelta(hours=24), valid_time, horizons],
        names=["forecast_origin", "valid_time", "horizon"],
    )
    return pd.DataFrame(
        {
            "actual": np.full(len(valid_time), 100.0),
            "lower": np.full(len(valid_time), 90.0),
            "upper": np.full(len(valid_time), 110.0),
            "horizon": horizons,
            "valid_time": valid_time,
        },
        index=index,
    )


def test_interval_evidence_rows_cover_methods_horizons_and_complete_dst_days():
    module = _module()
    ordinary_days = pd.date_range("2024-01-01T23:00Z", periods=23 * 24, freq="h", tz="UTC")
    spring = pd.date_range("2024-03-30T23:00Z", periods=23, freq="h", tz="UTC")
    fall = pd.date_range("2024-10-26T22:00Z", periods=25, freq="h", tz="UTC")
    intervals = _interval_frame(ordinary_days.append(spring).append(fall))
    metadata = {
        "stream_id": "hourly/point/residual_hybrid",
        "protocol_fingerprint": "fingerprint",
        "point_state_policy": "{\"fit_through\":\"2025-06-30\"}",
        "interval_state_policy": "{\"adaptive\":\"updates_after_actual_only\"}",
    }

    evidence = module._interval_evidence_rows(
        {"symmetric": intervals, "adaptive": intervals, "cqr": intervals},
        metadata=metadata,
        alpha=0.1,
    )

    assert set(evidence["method"]) == {"symmetric", "adaptive", "cqr"}
    assert set(evidence["slice_type"]) == {"aggregate", "horizon", "local_day"}
    assert set(range(1, 25)) <= set(evidence.loc[evidence["slice_type"] == "horizon", "slice_value"])
    local_days = evidence[evidence["slice_type"] == "local_day"]
    assert set(local_days["n"]) == {23, 24, 25}
    assert {
        "method",
        "level",
        "slice_type",
        "slice_value",
        "evaluation_period",
        "coverage_scope",
        "stream_id",
        "protocol_fingerprint",
        "nominal",
        "empirical_coverage",
        "mean_width",
        "interval_score",
        "n",
    } == set(evidence.columns)


def test_hourly_artifact_preserves_source_identity_after_interval_calculation():
    module = _module()
    source = _prediction_frame([1.0, 2.0])
    positional = source.reset_index(drop=True)[["actual", "prediction", "horizon"]]

    restored = module._restore_interval_identity(positional, source)
    artifact = module._hourly_artifact(
        restored,
        pd.DataFrame(index=source.index.get_level_values("valid_time")),
        weather_strategy="persistence",
    )

    assert restored.index.equals(source.index)
    assert artifact["forecast_origin"].tolist() == list(source.index.get_level_values("forecast_origin"))
    assert artifact["valid_time"].tolist() == list(source.index.get_level_values("valid_time"))
    assert artifact["weather_source_run"].eq("persistence").all()


def test_paired_daily_bootstrap_preserves_day_blocks():
    report = _module()._paired_daily_mae_bootstrap(
        {
            "reference": _prediction_frame([3.0, 3.0, 3.0]),
            "candidate": _prediction_frame([1.0, 1.0, 1.0]),
        },
        reference="reference",
        seed=42,
        n_bootstrap=100,
    )

    row = report.loc[("candidate", "reference")]
    assert row["mae_difference"] == -2.0
    assert row["ci_lower"] == -2.0
    assert row["ci_upper"] == -2.0
    assert row["probability_candidate_better"] == 1.0


def test_daily_anchor_is_fitted_once_before_calibration_and_reused(monkeypatch):
    module = _module()
    cfg = replace(Config.from_yaml(CONFIG), models={"lightgbm": {}})
    index = pd.Index(
        [
            pd.Timestamp("2019-08-30").date(),
            pd.Timestamp("2019-08-31").date(),
            pd.Timestamp("2019-09-01").date(),
            pd.Timestamp("2019-09-02").date(),
        ]
    )
    matrix = pd.DataFrame(
        {
            "daily_load": [100.0, 101.0, 102.0, 103.0],
            "x": [1.0, 2.0, 3.0, 4.0],
        },
        index=index,
    )

    class SpyEstimator:
        def __init__(self):
            self.fit_calls = 0
            self.training_values = None

        def fit(self, X, y):
            self.fit_calls += 1
            self.training_values = X.iloc[:, 0].to_numpy(copy=True)
            return self

        def predict(self, X):
            return 100.0 + X.iloc[:, 0].to_numpy()

    estimator = SpyEstimator()
    monkeypatch.setattr(module.pd, "read_parquet", lambda path: matrix)
    monkeypatch.setattr(module, "build_daily_features", lambda dataset, config: dataset)
    monkeypatch.setattr(module, "daily_exog_columns", lambda family: ["x"])
    monkeypatch.setattr(module, "daily_feature_matrix", lambda features, family: features)
    monkeypatch.setattr(module, "make_estimator", lambda kind, params: estimator)

    fitted, fitted_matrix, columns = module._fit_frozen_daily_anchor(cfg, "lightgbm")
    first = module._predict_daily_anchors(
        fitted,
        fitted_matrix,
        columns,
        start=index[2],
        end=index[2],
    )
    second = module._predict_daily_anchors(
        fitted,
        fitted_matrix,
        columns,
        start=index[3],
        end=index[3],
    )

    assert estimator.fit_calls == 1
    np.testing.assert_allclose(estimator.training_values, [1.0, 2.0])
    assert first.iloc[0] == 103.0
    assert second.iloc[0] == 104.0


def test_monthly_interval_evidence_is_complete_and_berlin_ordered():
    module = _module()
    valid_time = pd.to_datetime(
        [
            "2026-01-01T12:00Z",
            "2026-01-31T12:00Z",
            "2026-02-01T12:00Z",
            "2026-02-28T12:00Z",
        ]
    )
    intervals = _interval_frame(valid_time)
    metadata = {
        "stream_id": "hourly/point/residual_hybrid",
        "protocol_fingerprint": "fingerprint",
    }

    evidence = module._monthly_interval_evidence_rows(
        {"symmetric": intervals, "adaptive": intervals, "cqr": intervals},
        metadata=metadata,
        alpha=0.1,
    )

    assert list(evidence) == module._EVIDENCE_COLUMNS
    assert evidence["slice_type"].eq("month").all()
    assert evidence["slice_value"].tolist() == [
        "2026-01",
        "2026-02",
        "2026-01",
        "2026-02",
        "2026-01",
        "2026-02",
    ]
    assert set(evidence["method"]) == {"symmetric", "adaptive", "cqr"}
    assert len(evidence) == 6


def test_monthly_residual_drift_uses_positional_valid_time_and_protocol_identity():
    module = _module()
    valid_time = pd.to_datetime(
        [
            "2026-01-01T12:00Z",
            "2026-01-31T12:00Z",
            "2026-02-01T12:00Z",
            "2026-02-28T12:00Z",
        ]
    )
    test = pd.DataFrame(
        {
            "valid_time": valid_time,
            "error": [1.0, -3.0, 2.0, -6.0],
        }
    )
    monitor = pd.DataFrame(
        {
            "value": [1.0, 3.0, 2.0, 6.0],
            "running_mean": [1.0, 2.0, 2.0, 3.0],
            "statistic": [0.0, 2.0, 1.0, 5.0],
            "alert": [False, False, True, True],
        }
    )
    metadata = {
        "stream_id": "hourly/point/residual_hybrid",
        "protocol_fingerprint": "fingerprint",
    }

    report = module._monthly_residual_drift(test, monitor, metadata=metadata)

    assert list(report) == [
        "month",
        "evaluation_period",
        "monitoring_scope",
        "stream_id",
        "protocol_fingerprint",
        "mean_absolute_error_MW",
        "max_page_hinkley_statistic",
        "alert_count",
        "n",
    ]
    assert report["month"].tolist() == ["2026-01", "2026-02"]
    assert report["mean_absolute_error_MW"].tolist() == [2.0, 4.0]
    assert report["max_page_hinkley_statistic"].tolist() == [2.0, 5.0]
    assert report["alert_count"].tolist() == [0, 2]
    assert report["n"].sum() == len(monitor)
    assert report["evaluation_period"].eq("retrospective_final").all()
    assert report["monitoring_scope"].eq("prequential_monitoring").all()
    assert report["stream_id"].eq(metadata["stream_id"]).all()
    assert report["protocol_fingerprint"].eq(metadata["protocol_fingerprint"]).all()


def test_monthly_residual_drift_rejects_positional_length_mismatch():
    test = pd.DataFrame(
        {
            "valid_time": pd.to_datetime(["2026-01-01T12:00Z", "2026-01-02T12:00Z"]),
            "error": [1.0, 2.0],
        }
    )
    monitor = pd.DataFrame({"statistic": [0.0], "alert": [False]})

    with pytest.raises(ValueError, match="rows do not match"):
        _module()._monthly_residual_drift(
            test,
            monitor,
            metadata={"stream_id": "stream", "protocol_fingerprint": "fingerprint"},
        )


class _FrozenHourlyModel:
    feature_columns = ("origin_available",)

    def __init__(self):
        self.fit_calls = 0

    def fit(self, frame):
        self.fit_calls += 1
        return self

    def predict(self, frame):
        prediction = frame["origin_available"].to_numpy(dtype="float64") + 100.0
        index = pd.MultiIndex.from_frame(frame[["forecast_origin", "valid_time", "horizon"]])
        return pd.DataFrame({"prediction": prediction}, index=index)


def _hourly_runner_frame() -> pd.DataFrame:
    valid_time = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "forecast_origin": valid_time - pd.Timedelta(hours=24),
            "valid_time": valid_time,
            "horizon": 24,
            "hourly_load": np.arange(48, dtype="float64") + 100.0,
            "L_t-168": np.arange(48, dtype="float64") + 99.0,
            "origin_available": np.arange(48, dtype="float64"),
            "post_origin_weather": np.arange(48, dtype="float64") + 10.0,
        }
    )


def test_post_origin_weather_mutation_cannot_change_hourly_selection():
    module = _module()
    baseline = _hourly_runner_frame()
    mutated = baseline.copy()
    mutated["post_origin_weather"] += 10_000.0
    model = _FrozenHourlyModel().fit(baseline)

    first = module._predict_period(
        model,
        baseline,
        start=pd.Timestamp("2024-01-01").date(),
        end=pd.Timestamp("2024-01-02").date(),
    )
    second = module._predict_period(
        model,
        mutated,
        start=pd.Timestamp("2024-01-01").date(),
        end=pd.Timestamp("2024-01-02").date(),
    )

    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(
        module._model_ablation({"model": first}),
        module._model_ablation({"model": second}),
    )


def test_oracle_main_reuses_frozen_selection_and_isolates_outputs(tmp_path, monkeypatch):
    module = _module()
    config_text = CONFIG.read_text()
    replacements = {
        'raw_end: "2019-12-31"': 'raw_end: "2019-01-31"',
        'dataset_start: "2019-01-14"': 'dataset_start: "2019-01-01"',
        'train_end: "2019-06-30"': 'train_end: "2019-01-02"',
        'val_end: "2019-08-31"': 'val_end: "2019-01-05"',
        'calibration_start: "2019-09-01"': 'calibration_start: "2019-01-06"',
        'calibration_end: "2019-09-30"': 'calibration_end: "2019-01-07"',
        'test_start: "2019-10-01"': 'test_start: "2019-01-08"',
        'test_end: "2019-12-31"': 'test_end: "2019-01-09"',
        "models: { sarimax: { order: [2,1,1], seasonal_order: [1,0,1,7], refit: false, refit_period: 90 } }": "models: { random_forest: {}, lightgbm: {} }",
    }
    for old, new in replacements.items():
        config_text = config_text.replace(old, new)
    canonical_config = tmp_path / "canonical.yaml"
    oracle_config = tmp_path / "oracle.yaml"
    canonical_config.write_text(config_text)
    oracle_config.write_text(config_text.replace('weather_strategy: "persistence"', 'weather_strategy: "oracle"'))

    def local_days(start: str) -> pd.DatetimeIndex:
        first = pd.Timestamp(start, tz="Europe/Berlin").tz_convert("UTC")
        return pd.date_range(first, periods=48, freq="h")

    valid_time = local_days("2019-01-03").append(
        [local_days("2019-01-06"), local_days("2019-01-08")]
    )
    local_hour = valid_time.tz_convert("Europe/Berlin").hour.to_numpy(dtype="float64")
    dataset = pd.DataFrame(
        {
            "Temp_forecast": local_hour,
            "Wind_forecast": local_hour + 1.0,
            "HDD": np.maximum(18.0 - local_hour, 0.0),
            "CDD": np.maximum(local_hour - 18.0, 0.0),
            "L_t-24": 100.0 + local_hour,
            "L_t-168": 99.0 + local_hour,
        },
        index=valid_time,
    )
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    dataset.to_parquet(processed / "dataset_hourly.parquet")

    results = tmp_path / "results"
    (results / "metrics").mkdir(parents=True)
    pd.DataFrame({"MAE": [2.0, 1.0]}, index=["random_forest", "lightgbm"]).to_csv(
        results / "metrics" / "validation_metrics.csv"
    )
    (results / "validation_predictions").mkdir()
    pd.DataFrame(
        {"date": ["2019-01-03", "2019-01-04"], "forecast": [111.5, 111.5]}
    ).to_csv(results / "validation_predictions" / "lightgbm.csv", index=False)

    long_frame = pd.DataFrame(
        {
            "forecast_origin": valid_time - pd.Timedelta(hours=24),
            "valid_time": valid_time,
            "horizon": 24,
            "hourly_load": 100.0 + local_hour,
            "L_t-168": 99.0 + local_hour,
            "origin_available": local_hour,
        }
    )

    class FakeModel:
        feature_columns = ("origin_available",)

        def __init__(self, mode):
            self.mode = mode

        def predict(self, frame):
            actual = frame["hourly_load"].to_numpy(dtype="float64")
            if self.mode == "exact":
                prediction = actual
            elif self.mode == "reverse":
                prediction = 223.0 - actual
            else:
                prediction = np.full(len(frame), 111.5)
            index = pd.MultiIndex.from_frame(
                frame[["forecast_origin", "valid_time", "horizon"]]
            )
            return pd.DataFrame({"prediction": prediction}, index=index)

    def fit_models(cfg, frame, *, train_end):
        del frame, train_end
        modes = (
            ("reverse", "exact", "constant")
            if cfg.features["weather_strategy"] == "oracle"
            else ("exact", "reverse", "constant")
        )
        return dict(
            zip(
                ("residual_hybrid", "ridge_direct", "lightgbm_direct"),
                map(FakeModel, modes),
                strict=True,
            )
        )

    anchor_calls = []

    class Anchor:
        def predict(self, frame):
            return np.full(len(frame), 111.5)

    daily_matrix = pd.DataFrame(
        {"x": [1.0] * 4},
        index=pd.Index(
            [
                pd.Timestamp("2019-01-06").date(),
                pd.Timestamp("2019-01-07").date(),
                pd.Timestamp("2019-01-08").date(),
                pd.Timestamp("2019-01-09").date(),
            ]
        ),
    )

    def fit_anchor(cfg, kind):
        del cfg
        anchor_calls.append(kind)
        return Anchor(), daily_matrix, ("x",)

    def quantile_model(cfg, *, quantile):
        del cfg
        model = FakeModel("constant")
        model.fit = lambda frame: model
        model.predict = lambda frame: FakeModel("constant").predict(frame).assign(
            prediction=95.0 if quantile < 0.5 else 128.0
        )
        return model

    monkeypatch.setattr(module, "build_hourly_features", lambda frame, cfg: dataset)
    monkeypatch.setattr(module, "hourly_feature_matrix", lambda frame: frame)
    monkeypatch.setattr(module, "direct_horizon_frame", lambda frame, horizon: long_frame)
    monkeypatch.setattr(module, "_fit_models", fit_models)
    monkeypatch.setattr(module, "_fit_frozen_daily_anchor", fit_anchor)
    monkeypatch.setattr(module, "make_hourly_quantile_lightgbm", quantile_model)
    monkeypatch.setattr(module, "git_commit", lambda root: "a" * 40)

    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--config", str(canonical_config)])
    module.main()

    canonical = results / "hourly"
    selection = pd.read_csv(canonical / "model_selection.csv")
    assert selection.loc[0, ["selected_model", "daily_anchor_model"]].tolist() == [
        "residual_hybrid",
        "lightgbm",
    ]
    canonical_hashes = {
        path.relative_to(canonical): sha256_file(path)
        for path in canonical.rglob("*")
        if path.is_file()
    }
    result_files_before = {path for path in results.rglob("*") if path.is_file()}

    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--config", str(oracle_config)])
    module.main()

    oracle_root = results / "sensitivity" / "oracle"
    oracle_hourly = oracle_root / "hourly"
    oracle_ablation = pd.read_csv(
        oracle_hourly / "model_ablation_validation_reconciled.csv", index_col=0
    )
    oracle_protocol = json.loads((oracle_root / "evaluation_protocol.json").read_text())
    oracle_intervals = pd.read_csv(oracle_hourly / "test_intervals.csv")
    canonical_hashes_after = {
        path.relative_to(canonical): sha256_file(path)
        for path in canonical.rglob("*")
        if path.is_file()
    }
    new_result_files = {
        path for path in results.rglob("*") if path.is_file()
    } - result_files_before

    assert oracle_ablation.index[0] == "ridge_direct"
    assert oracle_intervals["stream_id"].eq("hourly/point/residual_hybrid").all()
    assert "hourly/anchor/lightgbm" in oracle_protocol["records"]
    assert anchor_calls == ["lightgbm", "lightgbm"]
    assert canonical_hashes_after == canonical_hashes
    assert new_result_files and all(path.is_relative_to(oracle_root) for path in new_result_files)
    assert not (oracle_hourly / "model_selection.csv").exists()


def test_hourly_point_and_quantile_fit_reused_across_calibration_and_final(monkeypatch):
    module = _module()
    cfg = Config.from_yaml(CONFIG)
    frame = _hourly_runner_frame()
    models = [_FrozenHourlyModel() for _ in range(3)]
    monkeypatch.setattr(module, "make_hourly_hybrid", lambda config: models[0])
    monkeypatch.setattr(module, "make_hourly_direct_ridge", lambda: models[1])
    monkeypatch.setattr(module, "make_hourly_direct_lightgbm", lambda config: models[2])

    fitted = module._fit_models(cfg, frame, train_end=pd.Timestamp("2024-01-01").date())
    for model in fitted.values():
        module._predict_period(
            model,
            frame,
            start=pd.Timestamp("2024-01-01").date(),
            end=pd.Timestamp("2024-01-01").date(),
        )
        module._predict_period(
            model,
            frame,
            start=pd.Timestamp("2024-01-02").date(),
            end=pd.Timestamp("2024-01-02").date(),
        )

    assert [model.fit_calls for model in models] == [1, 1, 1]


def test_missing_cqr_calibration_horizon_fails_before_output(tmp_path):
    calibration = pd.DataFrame(
        {
            "horizon": [1, 1],
            "actual": [100.0, 101.0],
            "lower_quantile": [90.0, 91.0],
            "upper_quantile": [110.0, 111.0],
        }
    )
    final = pd.DataFrame(
        {
            "horizon": [2],
            "actual": [102.0],
            "lower_quantile": [92.0],
            "upper_quantile": [112.0],
        }
    )
    output = tmp_path / "cqr_test_intervals.csv"

    with pytest.raises(ValueError, match="missing CQR calibration scores"):
        horizon_cqr_intervals(calibration, final, alpha=0.1)

    assert not output.exists()
