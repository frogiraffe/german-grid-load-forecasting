"""Generate row-aligned canonical daily comparison evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from loadfc.config import Config
from loadfc.evaluation.comparison import (
    assert_compatible_daily_artifacts,
    compare_daily_artifacts,
    compare_hourly_artifacts,
    paired_local_day_bootstrap,
)
from loadfc.evaluation.metrics import seasonal_naive_mae
from loadfc.evaluation.protocol import protocol_fingerprint
from loadfc.features.assemble import build_features


def _load_artifacts(directory: Path) -> dict[str, pd.DataFrame]:
    artifacts: dict[str, pd.DataFrame] = {}
    for path in sorted(directory.glob("*.csv")):
        artifacts[path.stem] = pd.read_csv(path)
    if not artifacts:
        raise ValueError(f"no prediction artifacts found in {directory}")
    return artifacts


def _load_hourly_artifacts(directory: Path, *, records: dict[str, dict[str, object]]) -> dict[str, pd.DataFrame]:
    artifacts: dict[str, pd.DataFrame] = {}
    for path in sorted(directory.glob("*.csv")):
        frame = pd.read_csv(path)
        if "prediction" not in frame or "actual" not in frame:
            continue
        required = {"actual", "prediction", "evaluation_period", "stream_id", "protocol_fingerprint"}
        if not required <= set(frame):
            raise ValueError(f"{path.name} hourly artifact is missing provenance columns")
        frame = frame.rename(columns={"prediction": "forecast"})
        name = path.stem
        if frame["evaluation_period"].nunique() != 1 or frame["evaluation_period"].iat[0] != "retrospective_final":
            raise ValueError(f"{name} artifact has invalid evaluation period")
        stream_id = str(frame["stream_id"].iat[0])
        if stream_id not in records:
            raise ValueError(f"{name} artifact stream is not registered: {stream_id}")
        if frame["protocol_fingerprint"].nunique() != 1 or str(frame["protocol_fingerprint"].iat[0]) != protocol_fingerprint(records[stream_id]):
            raise ValueError(f"{name} artifact protocol fingerprint mismatch")
        artifacts[name] = frame
    if not artifacts:
        raise ValueError(f"no hourly prediction artifacts found in {directory}")
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = Config.from_yaml(Path(args.config))
    results = cfg.path("results_dir")
    predictions = _load_artifacts(results / "predictions")
    manifest = json.loads((results / "evaluation_protocol.json").read_text())
    records = manifest.get("records", {})
    for name, frame in predictions.items():
        stream_id = f"daily/{name}"
        if stream_id not in records:
            raise ValueError(f"missing protocol record for {stream_id}")
        if frame.get("stream_id", pd.Series(dtype=object)).empty:
            raise ValueError(f"{name} artifact is missing protocol metadata")
        if str(frame["stream_id"].iloc[0]) != stream_id:
            raise ValueError(f"{name} artifact stream_id does not match its filename")
        if str(frame["protocol_fingerprint"].iloc[0]) != protocol_fingerprint(records[stream_id]):
            raise ValueError(f"{name} artifact protocol fingerprint mismatch")
    assert_compatible_daily_artifacts(predictions)
    dataset = pd.read_parquet(cfg.path("processed_dir") / "dataset.parquet")
    features = build_features(dataset, cfg)
    denominator = seasonal_naive_mae(
        features.loc[features.index < cfg.split.test_start, "daily_load"], season=7
    )
    output = compare_daily_artifacts(predictions, naive_mae=denominator)
    output.to_csv(results / "metrics" / "daily_comparison.csv", index=False)

    hourly_record = records.get("hourly/point/residual_hybrid")
    if hourly_record is None:
        raise ValueError("missing protocol record for hourly/point/residual_hybrid")
    hourly = _load_hourly_artifacts(results / "hourly" / "test_predictions", records=records)
    hourly_dataset = pd.read_parquet(cfg.path("processed_dir") / "dataset_hourly.parquet")
    hourly_denominator = seasonal_naive_mae(
        hourly_dataset.loc[hourly_dataset.index < pd.Timestamp(cfg.split.test_start, tz="UTC"), "hourly_load"],
        season=168,
    )
    hourly_output = compare_hourly_artifacts(hourly, naive_mae=hourly_denominator)
    selection_path = results / "hourly" / "model_selection.csv"
    selection = pd.read_csv(selection_path)
    if len(selection) != 1:
        raise ValueError("hourly model_selection.csv must contain exactly one row")
    selected_model = str(selection.iloc[0]["selected_model"])
    selection_metric = str(selection.iloc[0]["selection_metric"])
    hourly_output["validation_selection_metric"] = selection_metric
    hourly_output["validation_selected_model"] = selected_model
    hourly_output["validation_selection_evidence"] = "results/hourly/model_ablation_validation_reconciled.csv"
    hourly_output.to_csv(results / "hourly" / "hourly_comparison.csv", index=False)
    reference = selected_model if selected_model in hourly else "residual_hybrid"
    uncertainty = paired_local_day_bootstrap(
        hourly,
        reference=reference,
        selected_model=selected_model,
        seed=cfg.seed,
    )
    uncertainty["validation_selection_metric"] = selection_metric
    uncertainty["validation_selected_model"] = selected_model
    uncertainty["validation_selection_evidence"] = "results/hourly/model_ablation_validation_reconciled.csv"
    uncertainty.to_csv(results / "hourly" / "model_ablation_test_uncertainty.csv", index=False)
    print(f"wrote {len(output)} daily and {len(hourly_output)} hourly comparison rows")


if __name__ == "__main__":
    main()
