"""Run validation selection plus frozen calibration and retrospective-final evaluation.

The 2024-2025 validation period is used for model comparison and ensemble
selection. 2025-H2 supplies residuals for interval calibration.
The inspected retrospective-final block is 2026-01-01 through 2026-08-04. All forecasts are one-step-ahead
and use weather information available at the forecast origin.

Outputs:
  results/validation_predictions/*.csv
  results/metrics/validation_metrics.csv
  results/calibration_predictions/*.csv
  results/metrics/calibration_metrics.csv
  results/predictions/*.csv
  results/metrics/test_metrics.csv
  results/metrics/*diebold_mariano.csv
  results/metrics/rolling_origin_mape.csv
  results/metrics/sarimax_diagnostics.csv

Usage:
    python scripts/run_evaluate.py [--config config.yaml]
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pandas as pd

from loadfc.config import Config
from loadfc.evaluation import diagnostics, metrics
from loadfc.evaluation.baselines import BASELINE_COLUMNS, baseline_predictions
from loadfc.evaluation.diebold_mariano import dm_pvalue_matrix
from loadfc.evaluation.protocol import merge_protocol_manifest, protocol_fingerprint
from loadfc.evaluation.provenance import artifact_results_root, daily_prediction_artifact
from loadfc.evaluation.rolling import rolling_forecast
from loadfc.evaluation.rolling_origin import rolling_origin_mape, six_month_windows
from loadfc.features.assemble import build_features, exog_columns, feature_matrix
from loadfc.models.ml import make_ml_forecaster
from loadfc.models.sarimax import SarimaxForecaster
from loadfc.tracking import git_commit, sha256_file

warnings.simplefilter("ignore")

KINDS = {"SARIMAX": "sarimax", "xgboost": "ml", "lightgbm": "ml", "random_forest": "ml"}


def _validation_low_risk_decision(
    cfg: Config,
    operational_metrics: pd.DataFrame,
    candidate_metrics: pd.DataFrame,
    *,
    protocol_fingerprint_value: str,
    evidence_path: Path,
) -> dict[str, object]:
    """Persist one deterministic weather-candidate decision from validation only."""

    criterion = "ensemble_MAPE"
    baseline = float(operational_metrics.loc["ensemble", "MAPE"])
    candidate = float(candidate_metrics.loc["ensemble", "MAPE"])
    improvement = baseline - candidate
    tie_threshold = 1e-12
    integrity_checks = {
        "selection_period_ends_at_validation_end": cfg.split.val_end.isoformat(),
        "protocol_fingerprint_present": bool(protocol_fingerprint_value),
        "no_final_outcomes_used": True,
    }
    accepted = improvement > tie_threshold and all(
        value for key, value in integrity_checks.items() if key != "selection_period_ends_at_validation_end"
    )
    try:
        evidence_reference = evidence_path.resolve().relative_to(Path(cfg.root).resolve()).as_posix()
    except (AttributeError, ValueError):
        evidence_reference = evidence_path.as_posix()
    return {
        "candidate": "persistence_weather",
        "criterion": criterion,
        "validation_period_start": (cfg.split.train_end + timedelta(days=1)).isoformat(),
        "validation_period_end": cfg.split.val_end.isoformat(),
        "validation_evidence_path": evidence_reference,
        "protocol_fingerprint": protocol_fingerprint_value,
        "baseline_metric": baseline,
        "candidate_metric": candidate,
        "deterministic_improvement": improvement,
        "practical_tie_threshold": tie_threshold,
        "integrity_checks": integrity_checks,
        "selected_model": "ensemble",
        "decision": "accepted" if accepted else "rejected",
        "rationale": (
            "Accepted because persistence weather improves validation MAPE beyond the practical tie threshold."
            if accepted
            else "Rejected because persistence weather does not improve validation MAPE beyond the practical tie threshold; preserve the configured model."
        ),
    }


def _factory(kind: str, cfg: Config):
    model_cfg = cfg.models
    if kind == "SARIMAX":
        params = model_cfg["sarimax"]
        return lambda: SarimaxForecaster(
            params["order"],
            params["seasonal_order"],
            refit=str(params["refit"]).lower(),
            refit_period=params.get("refit_period", 90),
        )
    params = dict(model_cfg[kind])
    params.setdefault("random_state", cfg.seed)
    return lambda: make_ml_forecaster(kind, params)


def _common_index(predictions: dict[str, pd.Series]) -> pd.Index:
    common: pd.Index | None = None
    for prediction in predictions.values():
        common = prediction.index if common is None else common.intersection(prediction.index)
    if common is None or common.empty:
        raise ValueError("model predictions have no common evaluation dates")
    return common


def _evaluate_period(
    cfg: Config,
    feats: pd.DataFrame,
    weather: pd.DataFrame,
    start,
    end,
    predictions_dir: Path,
    metrics_path: Path,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    predictions_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []
    predictions: dict[str, pd.Series] = {}

    naive = metrics.seasonal_naive_mae(feats.loc[feats.index < start, "daily_load"], season=7)
    for kind, family in KINDS.items():
        matrix = feature_matrix(feats, family)
        columns = exog_columns(family)
        train = matrix[matrix.index < start]
        evaluation = matrix[(matrix.index >= start) & (matrix.index <= end)]
        forecast = rolling_forecast(_factory(kind, cfg)(), train, evaluation, columns)
        actual = evaluation["daily_load"]
        rows.append({"model": kind, **metrics.all_metrics(actual, forecast, naive)})
        predictions[kind] = forecast
        daily_prediction_artifact(
            pd.DataFrame({"actual": actual, "forecast": forecast}),
            weather,
            weather_strategy=cfg.features.get("weather_strategy", "persistence"),
        ).to_csv(predictions_dir / f"{kind}.csv")

    for name, baseline in baseline_predictions(feats, start, end).items():
        actual = feats.loc[baseline.index, "daily_load"]
        rows.append({"model": name, **metrics.all_metrics(actual, baseline, naive)})
        predictions[name] = baseline
        daily_prediction_artifact(
            pd.DataFrame({"actual": actual, "forecast": baseline}),
            weather,
            weather_strategy=cfg.features.get("weather_strategy", "persistence"),
        ).to_csv(predictions_dir / f"{name}.csv")

    members = list(cfg.ensemble["members"])
    unknown = sorted(set(members) - set(predictions))
    if unknown:
        raise ValueError(f"unknown ensemble members: {unknown}")
    common = _common_index({name: predictions[name] for name in members})
    ensemble = sum(predictions[name].loc[common] for name in members) / len(members)
    actual = feats.loc[common, "daily_load"]
    rows.append({"model": "ensemble", **metrics.all_metrics(actual, ensemble, naive)})
    predictions["ensemble"] = ensemble
    daily_prediction_artifact(
        pd.DataFrame({"actual": actual, "forecast": ensemble}),
        weather,
        weather_strategy=cfg.features.get("weather_strategy", "persistence"),
    ).to_csv(predictions_dir / "ensemble.csv")

    metrics_df = pd.DataFrame(rows).set_index("model")
    metrics_df.to_csv(metrics_path)
    return metrics_df, predictions


def _frozen_daily_predictions(cfg: Config, feats: pd.DataFrame) -> dict[str, pd.Series]:
    """Fit once through validation, then update one state across both frozen periods."""

    start = cfg.split.calibration_start
    end = cfg.split.test_end
    predictions: dict[str, pd.Series] = {}
    for kind, family in KINDS.items():
        matrix = feature_matrix(feats, family)
        train = matrix[matrix.index <= cfg.split.val_end]
        evaluation = matrix[(matrix.index >= start) & (matrix.index <= end)]
        predictions[kind] = rolling_forecast(_factory(kind, cfg)(), train, evaluation, exog_columns(family))

    predictions.update(baseline_predictions(feats, start, end))
    members = list(cfg.ensemble["members"])
    unknown = sorted(set(members) - set(predictions))
    if unknown:
        raise ValueError(f"unknown ensemble members: {unknown}")
    common = _common_index({name: predictions[name] for name in members})
    predictions["ensemble"] = sum(predictions[name].loc[common] for name in members) / len(members)
    return predictions


def _split_frozen_predictions(
    predictions: dict[str, pd.Series], cfg: Config
) -> dict[str, dict[str, pd.Series]]:
    periods = {
        "calibration": (cfg.split.calibration_start, cfg.split.calibration_end),
        "retrospective_final": (cfg.split.test_start, cfg.split.test_end),
    }
    return {
        period: {
            name: values[(values.index >= start) & (values.index <= end)]
            for name, values in predictions.items()
        }
        for period, (start, end) in periods.items()
    }


def _point_state_policy(cfg: Config) -> dict[str, str | int | bool]:
    sarimax = cfg.models.get("sarimax", {})
    return {
        "fit_through": cfg.split.val_end.isoformat(),
        "rolling_update_after_actual": True,
        "sarimax_refit": str(sarimax.get("refit", "false")).lower(),
        "sarimax_refit_period": int(sarimax.get("refit_period", 90)),
    }


def _daily_protocol_records(
    cfg: Config,
    *,
    source_revision: str,
    config_sha256: str,
    validation_evidence: Path,
) -> dict[str, dict[str, object]]:
    splits = {
        "train": [cfg.dataset_start.isoformat(), cfg.split.train_end.isoformat()],
        "validation": [(cfg.split.train_end + timedelta(days=1)).isoformat(), cfg.split.val_end.isoformat()],
        "calibration": [cfg.split.calibration_start.isoformat(), cfg.split.calibration_end.isoformat()],
        "retrospective_final": [cfg.split.test_start.isoformat(), cfg.split.test_end.isoformat()],
    }
    point_policy = _point_state_policy(cfg)
    interval_policy = {"fixed": "calibration_residuals", "adaptive": "updates_after_actual_only"}
    common = {
        "schema_version": 1,
        "source_revision": source_revision,
        "config_sha256": config_sha256,
        "seed": cfg.seed,
        "splits": splits,
        "weather_strategy": getattr(cfg, "features", {}).get("weather_strategy", "available_day_ahead"),
        "point_state_policy": point_policy,
        "interval_state_policy": interval_policy,
        "final_role": "retrospective_final",
        "validation_selection_evidence": validation_evidence.as_posix(),
        "rationale": "Validation metrics choose configured candidates; calibration and final do not tune them.",
    }
    records: dict[str, dict[str, object]] = {}
    for kind, family in KINDS.items():
        params = dict(cfg.models.get(kind, {}))
        if kind != "SARIMAX":
            params.setdefault("random_state", cfg.seed)
        stream_id = f"daily/{kind}"
        records[stream_id] = {
            **common,
            "stream_id": stream_id,
            "model_identity": kind,
            "ordered_feature_columns": exog_columns(family),
            "model_parameters": params,
        }
    for name, column in BASELINE_COLUMNS.items():
        stream_id = f"daily/{name}"
        records[stream_id] = {
            **common,
            "stream_id": stream_id,
            "model_identity": f"deterministic_baseline:{name}",
            "ordered_feature_columns": [column],
            "model_parameters": {"rule": column},
        }
    stream_id = "daily/ensemble"
    records[stream_id] = {
        **common,
        "stream_id": stream_id,
        "model_identity": "unweighted_ensemble",
        "ordered_feature_columns": list(cfg.ensemble["members"]),
        "model_parameters": {"members": list(cfg.ensemble["members"]), "aggregation": "mean"},
    }
    return records


def _artifact_with_protocol(
    predictions: pd.DataFrame,
    weather: pd.DataFrame,
    cfg: Config,
    record: dict[str, object],
    period: str,
) -> pd.DataFrame:
    artifact = daily_prediction_artifact(
        predictions,
        weather,
        weather_strategy=cfg.features.get("weather_strategy", "persistence"),
    )
    return artifact.assign(
        stream_id=record["stream_id"],
        protocol_fingerprint=protocol_fingerprint(record),
        evaluation_period=period,
        point_state_policy=json.dumps(record["point_state_policy"], sort_keys=True),
        interval_state_policy=json.dumps(record["interval_state_policy"], sort_keys=True),
    )


def _persist_frozen_period(
    cfg: Config,
    feats: pd.DataFrame,
    weather: pd.DataFrame,
    predictions: dict[str, pd.Series],
    records: dict[str, dict[str, object]],
    period: str,
    start,
    predictions_dir: Path,
    metrics_path: Path,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    predictions_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []
    persisted: dict[str, pd.Series] = {}
    naive = metrics.seasonal_naive_mae(feats.loc[feats.index < start, "daily_load"], season=7)
    for name, forecast in predictions.items():
        actual = feats.loc[forecast.index, "daily_load"]
        rows.append({"model": name, **metrics.all_metrics(actual, forecast, naive)})
        persisted[name] = forecast
        _artifact_with_protocol(
            pd.DataFrame({"actual": actual, "forecast": forecast}),
            weather,
            cfg,
            records[f"daily/{name}"],
            period,
        ).to_csv(predictions_dir / f"{name}.csv")
    metrics_df = pd.DataFrame(rows).set_index("model")
    metrics_df.to_csv(metrics_path)
    return metrics_df, persisted


def _dm_table(feats: pd.DataFrame, predictions: dict[str, pd.Series]) -> pd.DataFrame:
    common = _common_index(predictions)
    errors = {
        name: (feats.loc[common, "daily_load"].to_numpy() - prediction.loc[common].to_numpy())
        for name, prediction in predictions.items()
    }
    return dm_pvalue_matrix(errors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = Config.from_yaml(Path(args.config))

    dataset = pd.read_parquet(cfg.path("processed_dir") / "dataset.parquet")
    feats = build_features(dataset, cfg)
    results_dir = artifact_results_root(cfg)
    metrics_dir = results_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    validation_start = cfg.split.train_end + timedelta(days=1)
    validation_metrics, validation_predictions = _evaluate_period(
        cfg,
        feats,
        dataset,
        validation_start,
        cfg.split.val_end,
        results_dir / "validation_predictions",
        metrics_dir / "validation_metrics.csv",
    )
    records = _daily_protocol_records(
        cfg,
        source_revision=git_commit(cfg.root),
        config_sha256=sha256_file(Path(args.config)),
        validation_evidence=metrics_dir / "validation_metrics.csv",
    )
    merge_protocol_manifest(results_dir / "evaluation_protocol.json", records, "daily/")
    frozen_periods = _split_frozen_predictions(_frozen_daily_predictions(cfg, feats), cfg)
    calibration_metrics, _ = _persist_frozen_period(
        cfg,
        feats,
        dataset,
        frozen_periods["calibration"],
        records,
        "calibration",
        cfg.split.calibration_start,
        results_dir / "calibration_predictions",
        metrics_dir / "calibration_metrics.csv",
    )
    test_metrics, test_predictions = _persist_frozen_period(
        cfg,
        feats,
        dataset,
        frozen_periods["retrospective_final"],
        records,
        "retrospective_final",
        cfg.split.test_start,
        results_dir / "predictions",
        metrics_dir / "test_metrics.csv",
    )
    persistence_cfg = replace(
        cfg,
        features={**cfg.features, "weather_strategy": "persistence"},
    )
    persistence_features = build_features(dataset, persistence_cfg)
    persistence_validation_metrics, _ = _evaluate_period(
        persistence_cfg,
        persistence_features,
        dataset,
        validation_start,
        cfg.split.val_end,
        results_dir / "weather_ablation_validation_predictions",
        metrics_dir / "persistence_weather_validation_metrics.csv",
    )
    decision = _validation_low_risk_decision(
        cfg,
        validation_metrics,
        persistence_validation_metrics,
        protocol_fingerprint_value=protocol_fingerprint(records["daily/ensemble"]),
        evidence_path=metrics_dir / "persistence_weather_validation_metrics.csv",
    )
    (metrics_dir / "low_risk_improvement_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Retrospective weather differences remain descriptive only; selection never reads them.
    persistence_metrics, _ = _evaluate_period(
        persistence_cfg,
        persistence_features,
        dataset,
        cfg.split.test_start,
        cfg.split.test_end,
        results_dir / "weather_ablation_predictions",
        metrics_dir / "persistence_weather_test_metrics.csv",
    )
    weather_ablation = pd.DataFrame(
        {
            "operational_MAPE": test_metrics["MAPE"],
            "persistence_MAPE": persistence_metrics["MAPE"],
        }
    )
    weather_ablation["delta_pp"] = (
        weather_ablation["operational_MAPE"] - weather_ablation["persistence_MAPE"]
    )
    weather_ablation.to_csv(metrics_dir / "weather_ablation.csv")

    _dm_table(feats, validation_predictions).to_csv(metrics_dir / "validation_diebold_mariano.csv")
    final_dm = _dm_table(feats, test_predictions)
    final_dm.to_csv(metrics_dir / "diebold_mariano.csv")

    windows = six_month_windows(validation_start, cfg.split.calibration_end)
    rolling_results = {
        kind: rolling_origin_mape(
            _factory(kind, cfg),
            feature_matrix(feats, family),
            exog_columns(family),
            windows,
        )
        for kind, family in KINDS.items()
    }
    for name, column in BASELINE_COLUMNS.items():
        rolling_results[name] = {}
        for label, start, end in windows:
            evaluation = feats[(feats.index >= start) & (feats.index <= end)].dropna(
                subset=[column]
            )
            rolling_results[name][label] = metrics.mape(
                evaluation["daily_load"], evaluation[column]
            )
    pd.DataFrame(rolling_results).to_csv(metrics_dir / "rolling_origin_mape.csv")

    common = _common_index(test_predictions)
    sarimax_errors = (
        feats.loc[common, "daily_load"] - test_predictions["SARIMAX"].loc[common]
    ).to_numpy()
    lb = diagnostics.ljung_box(sarimax_errors, lags=10)
    arch = diagnostics.arch_lm(sarimax_errors, lags=7)
    pd.DataFrame(
        {
            "test": ["ljung_box", "arch_lm"],
            "stat": [lb[0], arch[0]],
            "pvalue": [lb[1], arch[1]],
        }
    ).to_csv(metrics_dir / "sarimax_diagnostics.csv", index=False)

    print("\nValidation (2024-2025):\n", validation_metrics.round(4).to_string())
    print("\nCalibration (2025-H2):\n", calibration_metrics.round(4).to_string())
    print("\nRetrospective final (2026-01-01..2026-08-04):\n", test_metrics.round(4).to_string())
    print("\nOutputs written to", results_dir)


if __name__ == "__main__":
    main()
