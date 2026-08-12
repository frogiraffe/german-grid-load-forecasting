"""Deterministic evaluation-protocol records and compatibility checks."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any

_REQUIRED_FIELDS = {
    "schema_version",
    "source_revision",
    "config_sha256",
    "stream_id",
    "model_identity",
    "ordered_feature_columns",
    "model_parameters",
    "seed",
    "splits",
    "weather_strategy",
    "point_state_policy",
    "interval_state_policy",
    "final_role",
    "validation_selection_evidence",
    "rationale",
}
_SPLIT_ROLES = {"train", "validation", "calibration", "retrospective_final"}
_ARTIFACT_FIELDS = {
    "stream_id",
    "protocol_fingerprint",
    "evaluation_period",
    "point_state_policy",
    "interval_state_policy",
}


def _normalized(value: Any) -> Any:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("protocol mapping keys must be strings")
        return {key: _normalized(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalized(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("protocol values must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"protocol value is not JSON-safe: {type(value).__name__}")


def validate_protocol_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize and validate a complete machine-readable protocol record."""

    missing = _REQUIRED_FIELDS - set(record)
    if missing:
        raise ValueError(f"protocol record missing required fields: {sorted(missing)}")
    normalized = _normalized(record)
    if not isinstance(normalized["stream_id"], str) or not normalized["stream_id"]:
        raise ValueError("protocol stream_id must be a non-empty string")
    if not isinstance(normalized["ordered_feature_columns"], list):
        raise ValueError("protocol ordered_feature_columns must be a list")
    if not isinstance(normalized["model_parameters"], dict):
        raise ValueError("protocol model_parameters must be a mapping")
    splits = normalized["splits"]
    if not isinstance(splits, dict) or set(splits) != _SPLIT_ROLES:
        raise ValueError("protocol splits must contain exactly the locked split roles")
    previous_end: date | None = None
    for role in ("train", "validation", "calibration", "retrospective_final"):
        bounds = splits[role]
        if not isinstance(bounds, list) or len(bounds) != 2 or not all(
            isinstance(value, str) for value in bounds
        ):
            raise ValueError(f"protocol {role} split must be a two-date range")
        start, end = (date.fromisoformat(value) for value in bounds)
        if start > end or (previous_end is not None and start <= previous_end):
            raise ValueError("protocol splits must be ordered and non-overlapping")
        previous_end = end
    return normalized


def canonical_protocol_json(record: Mapping[str, Any]) -> str:
    """Serialize a validated record with stable key ordering and compact JSON."""

    return json.dumps(
        validate_protocol_record(record),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def protocol_fingerprint(record: Mapping[str, Any]) -> str:
    """Return the SHA-256 compatibility fingerprint for a protocol record."""

    return hashlib.sha256(canonical_protocol_json(record).encode()).hexdigest()


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"schema_version": 1, "records": {}}
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), dict):
        raise ValueError("protocol manifest must contain a records mapping")
    return payload


def merge_protocol_manifest(
    path: Path,
    records: Mapping[str, Mapping[str, Any]],
    owner_prefix: str,
) -> dict[str, Any]:
    """Replace one owner namespace without mixing incompatible retained records."""

    if not owner_prefix:
        raise ValueError("protocol manifest owner prefix is required")
    normalized_records = {key: validate_protocol_record(value) for key, value in records.items()}
    if not normalized_records or any(not key.startswith(owner_prefix) for key in normalized_records):
        raise ValueError("protocol records must use the owner prefix")
    if any(value["stream_id"] != key for key, value in normalized_records.items()):
        raise ValueError("protocol manifest keys must match stream_id")

    manifest = _load_manifest(path)
    retained = {
        key: validate_protocol_record(value)
        for key, value in manifest["records"].items()
        if not key.startswith(owner_prefix)
    }
    reference = next(iter(normalized_records.values()))
    for existing in retained.values():
        for field in ("source_revision", "config_sha256", "splits"):
            if existing[field] != reference[field]:
                raise ValueError(f"protocol manifest conflict for {field}")

    merged = {**retained, **normalized_records}
    payload = {"schema_version": 1, "records": dict(sorted(merged.items()))}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
    return payload


def assert_compatible_artifacts(
    calibration: Mapping[str, Any],
    retrospective_final: Mapping[str, Any],
) -> None:
    """Fail before consumption unless two persisted stream fragments are compatible."""

    for name, artifact, expected_period in (
        ("calibration", calibration, "calibration"),
        ("retrospective_final", retrospective_final, "retrospective_final"),
    ):
        missing = _ARTIFACT_FIELDS - set(artifact)
        if missing:
            raise ValueError(f"{name} artifact missing required fields: {sorted(missing)}")
        if artifact["evaluation_period"] != expected_period:
            raise ValueError(f"{name} artifact has invalid evaluation_period")
    for field in _ARTIFACT_FIELDS - {"evaluation_period"}:
        if calibration[field] != retrospective_final[field]:
            raise ValueError(f"calibration/final protocol mismatch for {field}")
