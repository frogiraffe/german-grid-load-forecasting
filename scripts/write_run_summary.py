"""Write the compact JSON summary accompanying each evaluated release."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from loadfc import __version__
from loadfc.config import Config
from loadfc.evaluation.protocol import protocol_fingerprint, validate_protocol_record
from loadfc.presentation import MANAGED_PRESENTATION_PATHS
from loadfc.tracking import (
    PRESENTATION_ARTIFACTS as _PRESENTATION_ARTIFACTS,
)
from loadfc.tracking import (
    bundle_fingerprint as _bundle_fingerprint,
)
from loadfc.tracking import (
    git_commit,
    sha256_file,
    source_is_clean,
)
from loadfc.tracking import (
    source_root as _source_root,
)

_RESULT_SUFFIXES = {".csv", ".json", ".png"}


def _source_is_clean(source_root: Path, artifact_root: Path) -> bool:
    return source_is_clean(
        source_root,
        artifact_root,
        allowed_paths=(*MANAGED_PRESENTATION_PATHS, *_PRESENTATION_ARTIFACTS),
    )


def _canonical_result_files(results: Path) -> list[Path]:
    return [
        path
        for path in sorted(results.rglob("*"))
        if path.is_file()
        and path.name not in {"run_summary.json", "mlflow_run.csv"}
        and path.suffix in _RESULT_SUFFIXES
        and not any(
            part.startswith(".") or part == "__pycache__"
            for part in path.relative_to(results).parts
        )
    ]


def _release_identity(
    config_path: Path,
    results: Path,
    protocol_fingerprints: dict[str, str],
    *,
    artifact_root: Path | None = None,
) -> dict[str, object]:
    config_path = config_path.resolve()
    results = results.resolve()
    artifact_root = (artifact_root or config_path.parent).resolve()
    source_root = _source_root(config_path)
    lock_path = source_root / "uv.lock"
    if not lock_path.is_file():
        raise ValueError(f"release lockfile is missing: {lock_path}")

    artifact_hashes = {
        path.relative_to(results).as_posix(): sha256_file(path)
        for path in _canonical_result_files(results)
    }
    declared_artifacts: dict[str, str | None] = {
        f"results/{relative}": digest for relative, digest in artifact_hashes.items()
    }
    for relative in _PRESENTATION_ARTIFACTS:
        path = artifact_root / relative
        declared_artifacts[relative] = sha256_file(path) if path.is_file() else None
    declared_artifacts = dict(sorted(declared_artifacts.items()))

    identity: dict[str, object] = {
        "source_revision": git_commit(source_root),
        "source_clean": _source_is_clean(source_root, artifact_root),
        "config_sha256": sha256_file(config_path),
        "lock_sha256": sha256_file(lock_path),
        "protocol_fingerprints": protocol_fingerprints,
        "declared_artifacts": declared_artifacts,
    }
    identity["bundle_fingerprint"] = _bundle_fingerprint(identity)
    identity["artifact_hashes"] = artifact_hashes
    return identity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = Config.from_yaml(config_path)
    results = cfg.path("results_dir")

    validation = pd.read_csv(results / "metrics" / "validation_metrics.csv", index_col=0)
    calibration = pd.read_csv(results / "metrics" / "calibration_metrics.csv", index_col=0)
    test = pd.read_csv(results / "metrics" / "test_metrics.csv", index_col=0)
    weather_ablation = pd.read_csv(results / "metrics" / "weather_ablation.csv", index_col=0)
    manifest_path = results / "evaluation_protocol.json"
    manifest = json.loads(manifest_path.read_text())
    records = manifest.get("records") if isinstance(manifest, dict) else None
    if manifest.get("schema_version") != 1 or not isinstance(records, dict) or not records:
        raise ValueError("evaluation protocol manifest must contain registered records")
    fingerprints = {
        stream_id: protocol_fingerprint(validate_protocol_record(record))
        for stream_id, record in records.items()
    }
    dataset_path = cfg.path("processed_dir") / "dataset.csv"
    payload = {
        "record_type": "forecast_evaluation_summary",
        "schema_version": 2,
        "project_version": __version__,
        "target": "SMARD realised electricity consumption / grid load",
        "data": {
            "start": cfg.dataset_start.isoformat(),
            "end": cfg.raw_end.isoformat(),
            "rows": len(pd.read_csv(dataset_path)),
            "dataset_sha256": sha256_file(dataset_path),
        },
        "protocol": {
            "weather_strategy": cfg.features["weather_strategy"],
            "operational_weather_start": cfg.weather["operational_start"],
            "train_end": cfg.split.train_end.isoformat(),
            "validation_end": cfg.split.val_end.isoformat(),
            "calibration_start": cfg.split.calibration_start.isoformat(),
            "calibration_end": cfg.split.calibration_end.isoformat(),
            "test_start": cfg.split.test_start.isoformat(),
            "test_end": cfg.split.test_end.isoformat(),
            "ensemble_members": list(cfg.ensemble["members"]),
            "adaptive_gamma": float(cfg.uncertainty["adaptive_gamma"]),
            "adaptive_window": int(cfg.uncertainty["adaptive_window"]),
            "final_role": "retrospective_final",
        },
        "final_role": "retrospective_final",
        "protocol_manifest": manifest_path.name,
        "protocol_schema_version": 1,
        "runtime": {"python": platform.python_version()},
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "validation_metrics": validation.to_dict(orient="index"),
        "calibration_metrics": calibration.to_dict(orient="index"),
        "retrospective_final_metrics": test.to_dict(orient="index"),
        "test_metrics": test.to_dict(orient="index"),
        "weather_ablation": weather_ablation.to_dict(orient="index"),
    }
    payload.update(_release_identity(config_path, results, fingerprints))
    destination = results / "run_summary.json"
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("wrote", destination)


if __name__ == "__main__":
    main()
