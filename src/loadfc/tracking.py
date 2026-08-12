"""Opt-in local MLflow tracking and model registration."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

PRESENTATION_ARTIFACTS = (
    "report/generated_results.tex",
    "report/technical-report-en.pdf",
    "dashboard/src/generated/release.json",
)


@dataclass(frozen=True)
class TrackedModel:
    run_id: str
    model_name: str
    model_version: str
    tracking_uri: str


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for a local artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(root: Path) -> str:
    """Resolve the source revision without requiring GitPython."""

    if revision := os.environ.get("GITHUB_SHA"):
        return revision
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def source_root(path: Path) -> Path:
    """Resolve the Git top level containing ``path``."""

    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=path.parent,
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip()).resolve()


def source_is_clean(
    root: Path,
    artifact_root: Path | None = None,
    *,
    allowed_paths: Iterable[str | Path] = (),
) -> bool:
    """Return whether Git changes are limited to generated release outputs."""

    root = root.resolve()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    allowed = {Path(path).as_posix() for path in allowed_paths}
    staging_prefix = None
    if artifact_root is not None:
        artifact_root = artifact_root.resolve()
        if artifact_root != root and artifact_root.is_relative_to(root):
            staging_prefix = artifact_root.relative_to(root).as_posix().rstrip("/") + "/"
    return not [
        line
        for line in status
        if (path := line[3:].replace('"', "")) not in allowed
        and (staging_prefix is None or not path.startswith(staging_prefix))
    ]


def bundle_fingerprint(identity: Mapping[str, Any]) -> str:
    """Hash a release identity, excluding artifacts that embed the result."""

    fields = dict(identity)
    declared = fields.get("declared_artifacts")
    if isinstance(declared, dict):
        # Presentation artifacts embed this fingerprint, so their byte hashes are
        # verified separately but cannot be inputs to the fingerprint itself.
        declared = dict(declared)
        for relative in PRESENTATION_ARTIFACTS:
            if relative in declared:
                declared[relative] = None
        fields["declared_artifacts"] = declared
    encoded = json.dumps(fields, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def local_tracking_uri(root: Path) -> str:
    """Return the absolute SQLite URI used by the local registry."""

    return f"sqlite:///{(root / 'mlflow.db').resolve()}"


def track_sklearn_run(
    *,
    root: Path,
    experiment_name: str,
    run_name: str,
    model_name: str,
    estimator: Any,
    input_example: pd.DataFrame,
    params: Mapping[str, str | int | float | bool],
    metrics: Mapping[str, float],
    tags: Mapping[str, str],
    artifact_paths: Sequence[Path],
    tracking_uri: str | None = None,
) -> TrackedModel:
    """Log one fitted estimator and register it with a candidate alias."""

    uri = tracking_uri or local_tracking_uri(root)
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    artifact_root = (root / "mlartifacts").resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)

    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = client.create_experiment(
            experiment_name,
            artifact_location=artifact_root.as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id

    prediction = estimator.predict(input_example)
    signature = infer_signature(input_example, prediction)
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        mlflow.log_params(dict(params))
        mlflow.log_metrics(dict(metrics))
        mlflow.set_tags(dict(tags))
        for path in artifact_paths:
            mlflow.log_artifact(str(path), artifact_path="evaluation")
        mlflow.sklearn.log_model(
            sk_model=estimator,
            name="model",
            serialization_format="cloudpickle",
            signature=signature,
            input_example=input_example,
            registered_model_name=model_name,
            metadata={"feature_columns": list(input_example.columns)},
        )
        run_id = run.info.run_id

    versions = [
        version
        for version in client.search_model_versions(f"run_id = '{run_id}'")
        if version.name == model_name
    ]
    if len(versions) != 1:
        raise RuntimeError(f"expected one registered model version, found {len(versions)}")
    version = versions[0]
    client.set_registered_model_alias(model_name, "candidate", version.version)
    client.set_model_version_tag(
        model_name,
        version.version,
        "validation_status",
        "pending",
    )
    return TrackedModel(
        run_id=run_id,
        model_name=model_name,
        model_version=str(version.version),
        tracking_uri=uri,
    )
