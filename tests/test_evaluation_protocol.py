from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from loadfc.evaluation.protocol import (
    assert_compatible_artifacts,
    canonical_protocol_json,
    merge_protocol_manifest,
    protocol_fingerprint,
)


def _record(**changes):
    record = {
        "schema_version": 1,
        "source_revision": "abc123",
        "config_sha256": "config-sha",
        "stream_id": "daily/SARIMAX",
        "model_identity": "SARIMAX",
        "ordered_feature_columns": ["L_t-1", "Temp_forecast"],
        "model_parameters": {"order": [2, 1, 1], "refit": "false"},
        "seed": 42,
        "splits": {
            "train": ["2019-01-14", "2023-12-31"],
            "validation": ["2024-01-01", "2025-06-30"],
            "calibration": ["2025-07-01", "2025-12-31"],
            "retrospective_final": ["2026-01-01", "2026-08-04"],
        },
        "weather_strategy": "available_day_ahead",
        "point_state_policy": {"fit_through": "2025-06-30", "update": "after_actual"},
        "interval_state_policy": {"adaptive": "updates_after_actual_only"},
        "final_role": "retrospective_final",
        "validation_selection_evidence": "results/metrics/validation_metrics.csv",
        "rationale": "Validation metrics select the configured candidate; final is retrospective.",
    }
    record.update(changes)
    return record


def test_protocol_fingerprint_is_canonical_and_sensitive_to_ordered_schema():
    first = _record(model_parameters={"refit": "false", "order": [2, 1, 1]})
    second = _record(model_parameters={"order": [2, 1, 1], "refit": "false"})

    assert canonical_protocol_json(first) == canonical_protocol_json(second)
    assert protocol_fingerprint(first) == protocol_fingerprint(second)

    changed = deepcopy(first)
    changed["ordered_feature_columns"] = ["Temp_forecast", "L_t-1"]
    assert protocol_fingerprint(changed) != protocol_fingerprint(first)


def test_protocol_fingerprint_covers_every_frozen_input():
    baseline = _record()
    mutations = {
        "source_revision": "different-source",
        "config_sha256": "different-config",
        "model_parameters": {"order": [1, 0, 0], "refit": "true"},
        "seed": 7,
        "splits": {
            **baseline["splits"],
            "retrospective_final": ["2026-01-02", "2026-08-04"],
        },
        "weather_strategy": "persistence",
        "point_state_policy": {"fit_through": "2025-06-29", "update": "none"},
        "ordered_feature_columns": ["Temp_forecast", "L_t-1"],
    }

    fingerprints = {
        field: protocol_fingerprint(_record(**{field: value}))
        for field, value in mutations.items()
    }

    assert all(value != protocol_fingerprint(baseline) for value in fingerprints.values())
    assert len(set(fingerprints.values())) == len(fingerprints)


@pytest.mark.parametrize(
    "field",
    [
        "source_revision",
        "config_sha256",
        "model_parameters",
        "seed",
        "splits",
        "weather_strategy",
        "point_state_policy",
    ],
)
def test_protocol_rejects_missing_required_fields(field):
    record = _record()
    record.pop(field)

    with pytest.raises(ValueError, match="missing required"):
        protocol_fingerprint(record)


def test_manifest_replaces_only_its_owner_and_rejects_cross_owner_conflicts(tmp_path):
    path = tmp_path / "evaluation_protocol.json"
    stale_daily = _record(stream_id="daily/stale")
    hourly = _record(stream_id="hourly/residual")
    merge_protocol_manifest(path, {"daily/stale": stale_daily}, "daily/")
    merge_protocol_manifest(path, {"hourly/residual": hourly}, "hourly/")

    current_daily = _record(stream_id="daily/SARIMAX")
    manifest = merge_protocol_manifest(path, {"daily/SARIMAX": current_daily}, "daily/")

    assert set(manifest["records"]) == {"daily/SARIMAX", "hourly/residual"}

    conflicting = _record(stream_id="hourly/conflict", config_sha256="other-config")
    with pytest.raises(ValueError, match="config_sha256"):
        merge_protocol_manifest(path, {"hourly/conflict": conflicting}, "hourly/")


def test_calibration_and_final_artifacts_must_share_compatibility_metadata():
    record = _record()
    fingerprint = protocol_fingerprint(record)
    calibration = {
        "stream_id": record["stream_id"],
        "protocol_fingerprint": fingerprint,
        "evaluation_period": "calibration",
        "point_state_policy": record["point_state_policy"],
        "interval_state_policy": record["interval_state_policy"],
    }
    final = {**calibration, "evaluation_period": "retrospective_final"}

    assert_compatible_artifacts(calibration, final)
    final["protocol_fingerprint"] = "different"
    with pytest.raises(ValueError, match="protocol_fingerprint"):
        assert_compatible_artifacts(calibration, final)


def test_evaluation_ledger_records_the_frozen_chronology_and_evidence_limits():
    """Catch a ledger that omits a frozen boundary or overstates interval evidence."""

    ledger = (Path(__file__).parents[1] / "docs" / "EVALUATION_LEDGER.md").read_text()

    required_fragments = (
        "2019-01-14–2023-12-31",
        "2024-01-01–2025-06-30",
        "2025-07-01–2025-12-31",
        "2026-01-01–2026-08-04",
        "`retrospective_final`",
        "2024-01-01–2025-12-31",
        "2026-01-01–2026-06-30",
        "2026-07-01–2026-07-22",
        "already been inspected",
        "training/validation evidence only",
        "Calibration supplies interval residuals",
        "results/evaluation_protocol.json",
        "available_day_ahead",
        "Seed: `42`",
        "ordered feature schema",
        "point-state policy",
        "interval-state policy",
        "23-row threshold",
        "empirical retrospective coverage",
        "prequential monitoring only",
        "no unconditional time-series coverage guarantee",
    )

    assert all(fragment in ledger for fragment in required_fragments)
