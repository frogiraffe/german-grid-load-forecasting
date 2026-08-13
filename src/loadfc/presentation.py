"""Canonical values and managed blocks for published result summaries."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from loadfc.evaluation.protocol import protocol_fingerprint, validate_protocol_record
from loadfc.tracking import bundle_fingerprint as calculate_bundle_fingerprint

MARKDOWN_START = "<!-- loadfc:generated-start -->"
MARKDOWN_END = "<!-- loadfc:generated-end -->"
TEX_START = "% loadfc:generated-start"
TEX_END = "% loadfc:generated-end"
MANAGED_PRESENTATION_PATHS = (
    Path("README.md"),
    Path("report/technical-report-en.md"),
    Path("report/technical-report-en.tex"),
)
DAILY_PRESENTATION_MODEL = "ensemble"

_DISPLAY_NAMES = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "random_forest": "Random Forest",
    "naive_1d": "Naive (t-1)",
    "seasonal_naive_7d": "Seasonal naive (t-7)",
    "ensemble": "Ensemble",
    "residual_hybrid": "Residual hybrid",
    "lightgbm_direct": "Direct LightGBM",
    "ridge_direct": "Direct Ridge",
}
_UNQUALIFIED_CLAIMS = (
    re.compile(r"\bproduction(?:[- ]ready)?\b", re.IGNORECASE),
    re.compile(r"\b(?:live|real[- ]time)\s+(?:data|forecast|prediction|service|system)\b", re.IGNORECASE),
    re.compile(r"\bguarantee(?:d|s)?\b", re.IGNORECASE),
    re.compile(r"\bdistribution[- ]free\b", re.IGNORECASE),
    re.compile(r"\bcausal\s+(?:claim|effect|evidence|relationship)\b", re.IGNORECASE),
    re.compile(r"\bprospective\s+(?:claim|evaluation|evidence|holdout|result)\b", re.IGNORECASE),
    re.compile(r"\buntouched[- ]test\b", re.IGNORECASE),
)
_QUALIFIER = re.compile(
    r"\b(?:do(?:es)?\s+not|is\s+not|not|no|never|without|cannot|can't|requires?|should\s+not)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MetricEvidence:
    model: str
    mae: float
    rmse: float
    mape: float
    mase: float
    n: int


@dataclass(frozen=True)
class IntervalEvidence:
    method: str
    level: str
    nominal: float
    empirical_coverage: float
    mean_width: float
    interval_score: float
    n: int
    coverage_scope: str


@dataclass(frozen=True)
class PresentationValues:
    final_role: str
    final_role_label: str
    final_start: str
    final_end: str
    source_revision: str
    bundle_fingerprint: str
    daily_protocol_fingerprint: str
    hourly_protocol_fingerprint: str
    daily: MetricEvidence
    daily_baselines: tuple[MetricEvidence, ...]
    hourly_model: str
    daily_anchor_model: str
    hourly_validation_mae: float
    hourly_candidate_model: str
    hourly_candidate_validation_mae: float
    hourly_candidate_final_mae: float
    hourly_raw_mae: float
    hourly_reconciled_mae: float
    hourly_n: int
    bootstrap_difference: float
    bootstrap_lower: float
    bootstrap_upper: float
    bootstrap_probability: float
    bootstrap_n_days: int
    hourly_intervals: tuple[IntervalEvidence, ...]


def model_display_name(value: str) -> str:
    return _DISPLAY_NAMES.get(value, value)


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"missing presentation artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"presentation artifact must be a mapping: {path}")
    return payload


def _read_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise ValueError(f"missing presentation artifact: {path}")
    frame = pd.read_csv(path)
    missing = required - set(frame)
    if frame.empty or missing:
        raise ValueError(f"invalid presentation artifact {path}: missing {sorted(missing)}")
    return frame


def _finite(frame: pd.DataFrame, columns: tuple[str, ...], path: Path) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not values.map(math.isfinite).all():
            raise ValueError(f"non-finite presentation value {column}: {path}")


def _one(frame: pd.DataFrame, path: Path, description: str) -> pd.Series:
    if len(frame) != 1:
        raise ValueError(f"{description} must contain exactly one row: {path}")
    return frame.iloc[0]


def _metric(row: pd.Series) -> MetricEvidence:
    return MetricEvidence(
        model=str(row["model"]),
        mae=float(row["MAE"]),
        rmse=float(row["RMSE"]),
        mape=float(row["MAPE"]),
        mase=float(row["MASE"]),
        n=int(row["n"]),
    )


def load_presentation_values(results: Path) -> PresentationValues:
    """Load fail-closed presentation values from one canonical result bundle."""

    results = Path(results)
    manifest_path = results / "evaluation_protocol.json"
    manifest = _read_json(manifest_path)
    records_payload = manifest.get("records")
    if manifest.get("schema_version") != 1 or not isinstance(records_payload, dict) or not records_payload:
        raise ValueError(f"invalid protocol manifest: {manifest_path}")
    records = {
        str(stream_id): validate_protocol_record(record)
        for stream_id, record in records_payload.items()
        if isinstance(record, dict)
    }
    if len(records) != len(records_payload):
        raise ValueError(f"invalid protocol record: {manifest_path}")
    fingerprints = {
        stream_id: protocol_fingerprint(record) for stream_id, record in records.items()
    }
    manifest_identities = {
        (str(record["source_revision"]), str(record["config_sha256"]))
        for record in records.values()
    }
    if len(manifest_identities) != 1:
        raise ValueError(f"protocol manifest mixes source or config identities: {manifest_path}")
    source_revision, _ = manifest_identities.pop()

    daily_model = DAILY_PRESENTATION_MODEL

    daily_path = results / "metrics/daily_comparison.csv"
    daily = _read_csv(
        daily_path,
        {
            "model",
            "evaluation_period",
            "stream_id",
            "protocol_fingerprint",
            "start",
            "end",
            "n",
            "RMSE",
            "MAE",
            "MAPE",
            "MASE",
        },
    )
    _finite(daily, ("n", "RMSE", "MAE", "MAPE", "MASE"), daily_path)
    daily_n = pd.to_numeric(daily["n"])
    if not (daily_n.gt(0) & daily_n.mod(1).eq(0)).all():
        raise ValueError(f"invalid daily sample size: {daily_path}")
    if not daily["evaluation_period"].eq("retrospective_final").all():
        raise ValueError(f"daily comparison must use retrospective_final: {daily_path}")
    selected_daily = _one(
        daily[daily["model"].astype(str).eq(daily_model)], daily_path, "daily selected model"
    )
    baselines = daily[daily["model"].astype(str).isin({"naive_1d", "seasonal_naive_7d"})]
    if len(baselines) != 2 or set(baselines["model"].astype(str)) != {
        "naive_1d",
        "seasonal_naive_7d",
    }:
        raise ValueError(f"daily baseline evidence is incomplete: {daily_path}")

    selection_path = results / "hourly/model_selection.csv"
    selection = _one(
        _read_csv(
            selection_path,
            {
                "selected_model",
                "daily_anchor_model",
                "selection_evidence_period",
                "selection_protocol_fingerprint",
            },
        ),
        selection_path,
        "hourly selection",
    )
    if selection["selection_evidence_period"] != "validation":
        raise ValueError(f"hourly selection must use validation evidence: {selection_path}")
    hourly_model = str(selection["selected_model"])

    validation_path = results / "hourly/model_ablation_validation_reconciled.csv"
    validation = _read_csv(validation_path, {"model", "hourly_MAE"})
    _finite(validation, ("hourly_MAE",), validation_path)
    selected_validation = _one(
        validation[validation["model"].astype(str).eq(hourly_model)],
        validation_path,
        "hourly selected validation model",
    )

    bootstrap_path = results / "hourly/model_ablation_test_uncertainty.csv"
    bootstrap_rows = _read_csv(
        bootstrap_path,
        {
            "candidate",
            "reference",
            "mae_difference",
            "ci_lower",
            "ci_upper",
            "probability_candidate_better",
            "n_days",
            "selected_model",
            "selected_model_preserved",
            "validation_selected_model",
        },
    )
    bootstrap = _one(
        bootstrap_rows[bootstrap_rows["candidate"].astype(str).eq("lightgbm_direct")],
        bootstrap_path,
        "hourly LightGBM bootstrap comparison",
    )
    _finite(
        pd.DataFrame([bootstrap]),
        ("mae_difference", "ci_lower", "ci_upper", "probability_candidate_better", "n_days"),
        bootstrap_path,
    )
    if (
        str(bootstrap["reference"]) != hourly_model
        or str(bootstrap["selected_model"]) != hourly_model
        or str(bootstrap["validation_selected_model"]) != hourly_model
        or str(bootstrap["selected_model_preserved"]).lower() != "true"
        or not 0.0 <= float(bootstrap["probability_candidate_better"]) <= 1.0
        or float(bootstrap["ci_lower"]) > float(bootstrap["ci_upper"])
        or float(bootstrap["n_days"]) <= 0
        or float(bootstrap["n_days"]) % 1 != 0
    ):
        raise ValueError(f"bootstrap evidence does not preserve validation selection: {bootstrap_path}")
    candidate_model = str(bootstrap["candidate"])
    candidate_validation = _one(
        validation[validation["model"].astype(str).eq(candidate_model)],
        validation_path,
        "hourly bootstrap candidate",
    )

    raw_path = results / "hourly/model_ablation_test.csv"
    reconciled_path = results / "hourly/model_ablation_test_reconciled.csv"
    raw = _read_csv(raw_path, {"model", "hourly_MAE", "n_hours"})
    reconciled = _read_csv(reconciled_path, {"model", "hourly_MAE", "n_hours"})
    _finite(raw, ("hourly_MAE", "n_hours"), raw_path)
    _finite(reconciled, ("hourly_MAE", "n_hours"), reconciled_path)
    selected_raw = _one(
        raw[raw["model"].astype(str).eq(hourly_model)], raw_path, "hourly raw selected model"
    )
    selected_reconciled = _one(
        reconciled[reconciled["model"].astype(str).eq(hourly_model)],
        reconciled_path,
        "hourly reconciled selected model",
    )
    candidate_final = _one(
        reconciled[reconciled["model"].astype(str).eq(candidate_model)],
        reconciled_path,
        "hourly final bootstrap candidate",
    )
    if int(selected_raw["n_hours"]) != int(selected_reconciled["n_hours"]):
        raise ValueError("hourly raw and reconciled sample sizes disagree")
    if (
        float(selected_raw["n_hours"]) <= 0
        or float(selected_raw["n_hours"]) % 1 != 0
    ):
        raise ValueError("hourly selected sample size is invalid")

    reconciliation_path = results / "hourly/reconciliation_metrics.csv"
    reconciliation = _read_csv(
        reconciliation_path,
        {
            "model",
            "MAE",
            "n_hours",
            "evaluation_period",
            "stream_id",
            "protocol_fingerprint",
        },
    )
    _finite(reconciliation, ("MAE", "n_hours"), reconciliation_path)
    if not reconciliation["evaluation_period"].eq("retrospective_final").all():
        raise ValueError(f"reconciliation must use retrospective_final: {reconciliation_path}")
    reconciled_rows = reconciliation["model"].astype(str).str.contains("reconciled")
    raw_reconciliation = _one(
        reconciliation[~reconciled_rows], reconciliation_path, "raw reconciliation evidence"
    )
    reconciled_reconciliation = _one(
        reconciliation[reconciled_rows], reconciliation_path, "reconciled evidence"
    )
    if not math.isclose(
        float(reconciled_reconciliation["MAE"]), float(selected_reconciled["hourly_MAE"])
    ) or not math.isclose(float(raw_reconciliation["MAE"]), float(selected_raw["hourly_MAE"])):
        raise ValueError("hourly reconciliation evidence disagrees with selected comparison")
    if int(reconciled_reconciliation["n_hours"]) != int(selected_reconciled["n_hours"]):
        raise ValueError("hourly reconciliation sample size disagrees with selected comparison")

    interval_path = results / "hourly/interval_comparison.csv"
    intervals = _read_csv(
        interval_path,
        {
            "method",
            "level",
            "slice_type",
            "evaluation_period",
            "coverage_scope",
            "stream_id",
            "protocol_fingerprint",
            "nominal",
            "empirical_coverage",
            "mean_width",
            "interval_score",
            "n",
        },
    )
    try:
        _finite(
            intervals,
            ("nominal", "empirical_coverage", "mean_width", "interval_score", "n"),
            interval_path,
        )
    except ValueError as error:
        raise ValueError(f"invalid hourly interval evidence: {error}") from error
    if not intervals["evaluation_period"].eq("retrospective_final").all():
        raise ValueError(f"hourly interval evidence must use retrospective_final: {interval_path}")
    if (
        not intervals["slice_type"].eq("aggregate").all()
        or len(intervals) != 3
        or set(intervals["method"].astype(str)) != {"symmetric", "adaptive", "cqr"}
    ):
        raise ValueError(f"hourly aggregate interval evidence is incomplete: {interval_path}")
    numeric_intervals = intervals[
        ["nominal", "empirical_coverage", "mean_width", "interval_score", "n"]
    ].apply(pd.to_numeric)
    if (
        not numeric_intervals["nominal"].between(0.0, 1.0, inclusive="neither").all()
        or not numeric_intervals["empirical_coverage"].between(0.0, 1.0).all()
        or not numeric_intervals[["mean_width", "interval_score"]].ge(0.0).all().all()
        or not (
            numeric_intervals["n"].gt(0)
            & numeric_intervals["n"].mod(1).eq(0)
        ).all()
    ):
        raise ValueError(f"invalid hourly interval values: {interval_path}")
    expected_scopes = {
        "adaptive": "prequential_monitoring_no_unconditional_time_series_coverage"
    }
    for row in intervals.itertuples(index=False):
        expected_scope = expected_scopes.get(str(row.method), "empirical_retrospective")
        if row.coverage_scope != expected_scope:
            raise ValueError(f"invalid interval coverage scope: {interval_path}")
        stream_id = str(row.stream_id)
        if stream_id not in records or str(row.protocol_fingerprint) != fingerprints[stream_id]:
            raise ValueError(f"interval protocol identity conflict: {interval_path}")

    daily_stream = f"daily/{daily_model}"
    hourly_stream = f"hourly/point/{hourly_model}"
    anchor_stream = f"daily/{selection['daily_anchor_model']}"
    if daily_stream not in records or hourly_stream not in records or anchor_stream not in records:
        raise ValueError("selected presentation models lack protocol records")
    daily_record = records[daily_stream]
    hourly_record = records[hourly_stream]
    anchor_record = records[anchor_stream]
    if (
        daily_record["final_role"] != "retrospective_final"
        or hourly_record["final_role"] != "retrospective_final"
        or anchor_record["final_role"] != "retrospective_final"
        or daily_record["splits"] != hourly_record["splits"]
        or daily_record["splits"] != anchor_record["splits"]
    ):
        raise ValueError("daily and hourly presentation periods disagree")
    final_start, final_end = daily_record["splits"]["retrospective_final"]
    if (
        selected_daily["start"] != final_start
        or selected_daily["end"] != final_end
        or selected_daily["stream_id"] != daily_stream
        or selected_daily["protocol_fingerprint"] != fingerprints[daily_stream]
        or selection["selection_protocol_fingerprint"] != fingerprints[hourly_stream]
    ):
        raise ValueError("daily presentation identity conflicts with canonical protocol")
    for row in daily.itertuples(index=False):
        stream_id = str(row.stream_id)
        record = records.get(stream_id)
        if (
            record is None
            or record["final_role"] != "retrospective_final"
            or str(row.protocol_fingerprint) != fingerprints[stream_id]
            or str(row.start) != record["splits"]["retrospective_final"][0]
            or str(row.end) != record["splits"]["retrospective_final"][1]
        ):
            raise ValueError(f"daily comparison identity conflict: {daily_path}")
    if set(reconciliation["stream_id"].astype(str)) != {hourly_stream} or set(
        reconciliation["protocol_fingerprint"].astype(str)
    ) != {fingerprints[hourly_stream]}:
        raise ValueError("hourly presentation identity conflicts with canonical protocol")
    summary_path = results / "run_summary.json"
    summary = _read_json(summary_path)
    if (
        summary.get("schema_version") != 2
        or summary.get("final_role") != "retrospective_final"
        or summary.get("source_revision") != source_revision
        or summary.get("protocol_fingerprints") != fingerprints
    ):
        raise ValueError(f"run summary presentation identity conflict: {summary_path}")
    bundle_identity = summary.get("bundle_fingerprint")
    declared_artifacts = summary.get("declared_artifacts")
    fingerprint_fields = {
        key: summary.get(key)
        for key in (
            "source_revision",
            "source_clean",
            "config_sha256",
            "lock_sha256",
            "protocol_fingerprints",
            "declared_artifacts",
        )
    }
    if (
        not isinstance(bundle_identity, str)
        or re.fullmatch(r"[0-9a-f]{64}", bundle_identity) is None
        or not isinstance(declared_artifacts, dict)
        or calculate_bundle_fingerprint(fingerprint_fields) != bundle_identity
    ):
        raise ValueError(f"run summary bundle fingerprint conflict: {summary_path}")

    interval_order = {"symmetric": 0, "adaptive": 1, "cqr": 2}
    interval_values = tuple(
        IntervalEvidence(
            method=str(row.method),
            level=str(row.level),
            nominal=float(row.nominal),
            empirical_coverage=float(row.empirical_coverage),
            mean_width=float(row.mean_width),
            interval_score=float(row.interval_score),
            n=int(row.n),
            coverage_scope=str(row.coverage_scope),
        )
        for row in sorted(intervals.itertuples(index=False), key=lambda item: interval_order[str(item.method)])
    )
    return PresentationValues(
        final_role="retrospective_final",
        final_role_label="Retrospective; previously inspected",
        final_start=str(final_start),
        final_end=str(final_end),
        source_revision=source_revision,
        bundle_fingerprint=bundle_identity,
        daily_protocol_fingerprint=fingerprints[daily_stream],
        hourly_protocol_fingerprint=fingerprints[hourly_stream],
        daily=_metric(selected_daily),
        daily_baselines=tuple(
            _metric(row) for _, row in baselines.sort_values("model").iterrows()
        ),
        hourly_model=hourly_model,
        daily_anchor_model=str(selection["daily_anchor_model"]),
        hourly_validation_mae=float(selected_validation["hourly_MAE"]),
        hourly_candidate_model=candidate_model,
        hourly_candidate_validation_mae=float(candidate_validation["hourly_MAE"]),
        hourly_candidate_final_mae=float(candidate_final["hourly_MAE"]),
        hourly_raw_mae=float(raw_reconciliation["MAE"]),
        hourly_reconciled_mae=float(reconciled_reconciliation["MAE"]),
        hourly_n=int(reconciled_reconciliation["n_hours"]),
        bootstrap_difference=float(bootstrap["mae_difference"]),
        bootstrap_lower=float(bootstrap["ci_lower"]),
        bootstrap_upper=float(bootstrap["ci_upper"]),
        bootstrap_probability=float(bootstrap["probability_candidate_better"]),
        bootstrap_n_days=int(bootstrap["n_days"]),
        hourly_intervals=interval_values,
    )


def _format_date(value: str) -> str:
    return date.fromisoformat(value).strftime("%d %b %Y")


def _markers(surface: str | Path) -> tuple[str, str, bool]:
    is_tex = str(surface).lower().endswith(".tex") or str(surface).lower() == "tex"
    return (TEX_START, TEX_END, True) if is_tex else (MARKDOWN_START, MARKDOWN_END, False)


def _interval_line(interval: IntervalEvidence) -> str:
    return (
        f"{interval.method.upper() if interval.method == 'cqr' else interval.method.title()} "
        f"{interval.level}: target {100 * interval.nominal:.0f}%, measured coverage "
        f"{100 * interval.empirical_coverage:.2f}%, mean width {interval.mean_width:.1f} MW, "
        f"interval score {interval.interval_score:.1f} MW, n={interval.n}"
    )


def _markdown_block(values: PresentationValues) -> str:
    baselines = "; ".join(
        f"{model_display_name(item.model)} MAPE {item.mape:.3f}% (n={item.n})"
        for item in values.daily_baselines
    )
    intervals = "; ".join(_interval_line(item) for item in values.hourly_intervals)
    change = 100.0 * (values.hourly_reconciled_mae / values.hourly_raw_mae - 1.0)
    return f"""### Release results

- **Period:** {_format_date(values.final_start)} to {_format_date(values.final_end)}; n={values.daily.n} days.
- **Daily forecast:** {model_display_name(values.daily.model)}. MAE {values.daily.mae:.1f} MW, MAPE {values.daily.mape:.3f}%, and MASE {values.daily.mase:.3f}. Reference models: {baselines}.
- **Hourly forecast:** {model_display_name(values.hourly_model)}. Daily-total alignment reduced MAE from {values.hourly_raw_mae:.1f} MW to {values.hourly_reconciled_mae:.1f} MW ({change:+.2f}%). The result contains n={values.hourly_n} hourly values. The daily model was {model_display_name(values.daily_anchor_model)}.
- **Model comparison:** Validation MAE was {values.hourly_validation_mae:.1f} MW for {model_display_name(values.hourly_model)} and {values.hourly_candidate_validation_mae:.1f} MW for {model_display_name(values.hourly_candidate_model)}. The paired difference on the final data was {values.bootstrap_difference:+.1f} MW. Its 95% range was [{values.bootstrap_lower:.1f}, {values.bootstrap_upper:.1f}] MW across {values.bootstrap_n_days} days. This range includes zero, so the data do not show a clear winner.
- **Uncertainty ranges:** {intervals}.
<!-- provenance: source `{values.source_revision}`; daily protocol `{values.daily_protocol_fingerprint}`; hourly protocol `{values.hourly_protocol_fingerprint}`; bundle `{values.bundle_fingerprint}` -->
- **Scope:** The final period was already examined. These results describe this data period and do not state future accuracy.
"""


def _latex(value: object) -> str:
    return (
        str(value).replace("**", "").replace("`", "")
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def _tex_block(values: PresentationValues) -> str:
    markdown = _markdown_block(values).splitlines()
    items = [line.removeprefix("- ") for line in markdown if line.startswith("- ")]
    rendered = "\n".join(f"  \\item {_latex(item)}" for item in items)
    return f"""\\subsection*{{Release results}}
\\begin{{itemize}}
{rendered}
\\end{{itemize}}
% provenance: source {values.source_revision}; daily protocol {values.daily_protocol_fingerprint}; hourly protocol {values.hourly_protocol_fingerprint}; bundle {values.bundle_fingerprint}
"""


def render_managed_presentation(values: PresentationValues, surface: str | Path) -> str:
    """Render the exact marker-delimited block for a Markdown or TeX surface."""

    start, end, is_tex = _markers(surface)
    body = _tex_block(values) if is_tex else _markdown_block(values)
    return f"{start}\n{body}{end}"


def _parts(text: str, surface: str | Path) -> tuple[str, str]:
    start, end, _ = _markers(surface)
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"managed presentation requires exactly one marker pair: {surface}")
    start_at = text.index(start)
    end_at = text.index(end, start_at + len(start))
    return text[:start_at], text[end_at + len(end) :]


def _check_qualified_claims(text: str, surface: str | Path) -> None:
    for sentence in re.split(r"[.!?\n]+", text):
        if _QUALIFIER.search(sentence):
            continue
        if any(pattern.search(sentence) for pattern in _UNQUALIFIED_CLAIMS):
            raise ValueError(f"unqualified prohibited claim in {surface}: {sentence.strip()}")


def replace_managed_presentation(
    text: str, values: PresentationValues, surface: str | Path
) -> str:
    """Replace only the managed block while preserving all authored bytes around it."""

    before, after = _parts(text, surface)
    _check_qualified_claims(before + after, surface)
    return before + render_managed_presentation(values, surface) + after


def check_managed_presentation(
    text: str, values: PresentationValues, surface: str | Path
) -> None:
    """Reject missing, duplicated, stale, edited, or unqualified presentation content."""

    expected = replace_managed_presentation(text, values, surface)
    if text != expected:
        raise ValueError(f"managed presentation drift: {surface}")
