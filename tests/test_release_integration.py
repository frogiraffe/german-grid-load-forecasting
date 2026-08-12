from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from loadfc.evaluation.protocol import protocol_fingerprint
from loadfc.presentation import MANAGED_PRESENTATION_PATHS
from loadfc.tracking import PRESENTATION_ARTIFACTS, sha256_file
from scripts import run_pipeline
from scripts.validate_results import _source_is_clean, _validate_release_manifest
from scripts.write_run_summary import _release_identity

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_pipeline.py"
PRESENTATION_PATHS = (
    *MANAGED_PRESENTATION_PATHS,
    Path("report/generated_results.tex"),
    Path("report/technical-report-en.pdf"),
)


def _module():
    spec = importlib.util.spec_from_file_location("run_pipeline", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Release Test")
    _git(repo, "config", "user.email", "release@example.invalid")
    for directory in ("data/raw", "data/processed", "results"):
        (repo / directory).mkdir(parents=True)
    (repo / "data/raw/input.csv").write_text("raw\n")
    (repo / "data/processed/dataset.csv").write_text("processed\n")
    (repo / "results/previous.txt").write_text("previous canonical bundle\n")
    (repo / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "paths": {
                    "raw_dir": "data/raw",
                    "processed_dir": "data/processed",
                    "results_dir": "results",
                }
            },
            sort_keys=False,
        )
    )
    (repo / "uv.lock").write_text("version = 1\n")
    dashboard = repo / "dashboard/src/generated/release.json"
    dashboard.parent.mkdir(parents=True)
    dashboard.write_text("previous dashboard\n")
    for relative in PRESENTATION_PATHS:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"previous {relative.as_posix()}\n")
    (repo / ".gitignore").write_text("/results/\n")
    _git(
        repo,
        "add",
        ".gitignore",
        "README.md",
        "config.yaml",
        "dashboard",
        "data",
        "docs",
        "report",
        "uv.lock",
    )
    _git(repo, "commit", "-qm", "fixture")
    return repo


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _fake_compile_report(staging_root: Path) -> None:
    path = staging_root / "report/technical-report-en.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"staged pdf\n")


def _fake_run_summary(results: Path) -> None:
    (results / "run_summary.json").write_text(
        json.dumps({"bundle_fingerprint": "a" * 64})
    )


def test_report_compile_is_byte_stable_across_caller_epochs(tmp_path, monkeypatch):
    module = _module()
    repo = _repo(tmp_path)
    compiler_dir = tmp_path / "bin"
    compiler_dir.mkdir()
    compiler = compiler_dir / "tectonic"
    compiler.write_text(
        f"#!{sys.executable}\n"
        "import os\n"
        "from pathlib import Path\n"
        'Path("report/technical-report-en.pdf").write_text('
        'os.environ.get("SOURCE_DATE_EPOCH", "missing"))\n'
    )
    compiler.chmod(0o755)
    monkeypatch.setenv("PATH", str(compiler_dir))

    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1")
    module._compile_report(repo)
    first = (repo / "report/technical-report-en.pdf").read_bytes()

    monkeypatch.setenv("SOURCE_DATE_EPOCH", "2")
    module._compile_report(repo)
    second = (repo / "report/technical-report-en.pdf").read_bytes()

    assert first == second
    assert first != b"missing"


def test_dirty_source_rejection_is_read_only(tmp_path, monkeypatch):
    module = _module()
    repo = _repo(tmp_path)
    dirty = repo / "notes.txt"
    dirty.write_text("keep me\n")
    status_before = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    digest_before = _tree_digest(repo)
    calls: list[list[str]] = []
    real_run = subprocess.run

    def recording_run(command, **kwargs):
        calls.append(list(command))
        return real_run(command, **kwargs)

    monkeypatch.setattr(module.subprocess, "run", recording_run)

    with pytest.raises(RuntimeError, match="clean committed source"):
        module._assert_clean_source(repo)

    assert _tree_digest(repo) == digest_before
    assert _git(repo, "status", "--porcelain=v1", "--untracked-files=all") == status_before
    assert all("clean" not in command and "reset" not in command for command in calls)


@pytest.mark.parametrize(
    ("failing_stage", "occurrence"),
    [
        ("scripts/run_evaluate.py", 1),
        ("scripts/write_run_summary.py", 1),
        ("scripts/render_report_data.py", 1),
        ("tectonic", 1),
        ("scripts/write_run_summary.py", 2),
        ("scripts/validate_results.py", 1),
    ],
)
def test_staging_failure_preserves_previous_bundle(
    tmp_path, monkeypatch, failing_stage, occurrence
):
    module = _module()
    repo = _repo(tmp_path)
    monkeypatch.setattr(module, "ROOT", repo)
    before = _tree_digest(repo / "results")
    counts: dict[str, int] = {}

    def fake_run(*arguments: str) -> None:
        script = arguments[0]
        counts[script] = counts.get(script, 0) + 1
        config = Path(arguments[arguments.index("--config") + 1])
        paths = yaml.safe_load(config.read_text())["paths"]
        staged_results = (config.parent / paths["results_dir"]).resolve()
        staged_results.mkdir(parents=True, exist_ok=True)
        (staged_results / f"{Path(script).stem}.txt").write_text("staged\n")
        if script == "scripts/write_run_summary.py":
            _fake_run_summary(staged_results)
        if script == failing_stage and counts[script] == occurrence:
            raise subprocess.CalledProcessError(1, script)

    def fake_compile(staging_root: Path) -> None:
        if failing_stage == "tectonic":
            raise subprocess.CalledProcessError(1, "tectonic")
        _fake_compile_report(staging_root)

    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(module, "_compile_report", fake_compile)

    with pytest.raises(subprocess.CalledProcessError):
        module._run_release(repo / "config.yaml", refresh_data=False)

    assert _tree_digest(repo / "results") == before
    assert not list(repo.glob(".results-staging-*"))


def test_clean_release_runs_all_stages_in_staging_and_promotes_once(tmp_path, monkeypatch):
    module = _module()
    repo = _repo(tmp_path)
    monkeypatch.setattr(module, "ROOT", repo)
    original_config = (repo / "config.yaml").read_bytes()
    calls: list[tuple[str, Path]] = []
    promotions = 0
    canonical_dashboard = repo / "dashboard/src/generated/release.json"
    real_promote = module._promote_release

    def fake_run(*arguments: str) -> None:
        script = arguments[0]
        config = Path(arguments[arguments.index("--config") + 1])
        assert config.read_bytes() == original_config
        paths = yaml.safe_load(config.read_text())["paths"]
        staged_results = (config.parent / paths["results_dir"]).resolve()
        staged_results.mkdir(parents=True, exist_ok=True)
        calls.append((script, config))
        if script == "scripts/validate_results.py":
            assert len(list(staged_results.glob("*.txt"))) == len(module._RELEASE_STAGES)
        elif script == "scripts/build_dashboard_data.py":
            staged_dashboard = config.parent / "dashboard/src/generated/release.json"
            staged_dashboard.parent.mkdir(parents=True)
            staged_dashboard.write_text("new dashboard\n")
        else:
            (staged_results / f"{len(calls):02d}-{Path(script).stem}.txt").write_text("staged\n")
            if script == "scripts/write_run_summary.py":
                _fake_run_summary(staged_results)
            if script == "scripts/render_report_data.py":
                for relative in module._RELEASE_PRESENTATION_PATHS:
                    path = config.parent / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(f"new {relative.as_posix()}\n")

    def recording_promote(
        staged: Path,
        canonical: Path,
        staged_dashboard: Path,
        dashboard: Path,
        presentation_files: tuple[tuple[Path, Path], ...],
    ) -> None:
        nonlocal promotions
        promotions += 1
        real_promote(staged, canonical, staged_dashboard, dashboard, presentation_files)

    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(module, "_promote_release", recording_promote)
    monkeypatch.setattr(module, "_compile_report", _fake_compile_report)

    module._run_release(repo / "config.yaml", refresh_data=False)

    assert [script for script, _ in calls] == [
        *module._RELEASE_STAGES,
        "scripts/validate_results.py",
        "scripts/build_dashboard_data.py",
        "scripts/validate_results.py",
    ]
    assert all(config != repo / "config.yaml" for _, config in calls)
    assert len({config for _, config in calls}) == 1
    assert (repo / "config.yaml").read_bytes() == original_config
    assert not (repo / "results/previous.txt").exists()
    assert canonical_dashboard.read_text() == "new dashboard\n"
    for relative in PRESENTATION_PATHS:
        expected = (
            "staged pdf\n"
            if relative == Path("report/technical-report-en.pdf")
            else f"new {relative.as_posix()}\n"
        )
        assert (repo / relative).read_text() == expected
    assert promotions == 1
    assert not list(repo.glob(".results-staging-*"))


def test_failed_promotion_restores_previous_bundle(tmp_path, monkeypatch):
    module = _module()
    repo = _repo(tmp_path)
    staged = repo / ".results-staging-test" / "results"
    staged.mkdir(parents=True)
    (staged / "new.txt").write_text("new\n")
    before = _tree_digest(repo / "results")
    real_replace = Path.replace

    def fail_staged_replace(path: Path, target: Path):
        if path == staged:
            raise OSError("injected promotion failure")
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_staged_replace)

    with pytest.raises(OSError, match="injected promotion failure"):
        module._promote_bundle(staged, repo / "results")

    assert _tree_digest(repo / "results") == before
    assert not (repo / ".results-backup").exists()


@pytest.mark.parametrize("failing_destination", ["results", "dashboard", "presentation"])
def test_coordinated_promotion_restores_every_previous_destination(
    tmp_path, monkeypatch, failing_destination
):
    module = _module()
    repo = _repo(tmp_path)
    canonical_results = repo / "results"
    canonical_dashboard = repo / "dashboard/src/generated/release.json"
    canonical_dashboard.write_text("previous dashboard\n")
    staged_root = repo / ".results-staging-test"
    staged_results = staged_root / "results"
    staged_results.mkdir(parents=True)
    (staged_results / "new.txt").write_text("new results\n")
    staged_dashboard = staged_root / "dashboard/src/generated/release.json"
    staged_dashboard.parent.mkdir(parents=True)
    staged_dashboard.write_text("new dashboard\n")
    presentation_files = []
    presentation_before = {}
    for relative in PRESENTATION_PATHS:
        canonical = repo / relative
        staged_path = staged_root / relative
        staged_path.parent.mkdir(parents=True, exist_ok=True)
        staged_path.write_text(f"new {relative.as_posix()}\n")
        presentation_files.append((staged_path, canonical))
        presentation_before[canonical] = canonical.read_bytes()
    results_before = _tree_digest(canonical_results)
    dashboard_before = canonical_dashboard.read_bytes()
    real_replace = Path.replace

    def fail_selected_replace(path: Path, target: Path):
        if failing_destination == "results" and path == staged_results:
            raise OSError("injected results promotion failure")
        if failing_destination == "dashboard" and path == staged_dashboard:
            raise OSError("injected dashboard promotion failure")
        if failing_destination == "presentation" and path == presentation_files[0][0]:
            raise OSError("injected presentation promotion failure")
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_selected_replace)

    with pytest.raises(OSError, match="injected"):
        module._promote_release(
            staged_results,
            canonical_results,
            staged_dashboard,
            canonical_dashboard,
            tuple(presentation_files),
        )

    assert _tree_digest(canonical_results) == results_before
    assert canonical_dashboard.read_bytes() == dashboard_before
    assert {path: path.read_bytes() for path in presentation_before} == presentation_before
    assert not (repo / ".results-backup").exists()
    assert not (canonical_dashboard.parent / ".release.json-backup").exists()
    assert not any(
        path.with_name(f".{path.name}-backup").exists() for path in presentation_before
    )


@pytest.mark.parametrize(
    ("failing_script", "occurrence"),
    [
        ("scripts/validate_results.py", 1),
        ("scripts/build_dashboard_data.py", 1),
        ("scripts/validate_results.py", 2),
    ],
)
def test_release_gate_failure_preserves_canonical_outputs(
    tmp_path, monkeypatch, failing_script, occurrence
):
    module = _module()
    repo = _repo(tmp_path)
    monkeypatch.setattr(module, "ROOT", repo)
    dashboard = repo / "dashboard/src/generated/release.json"
    results_before = _tree_digest(repo / "results")
    dashboard_before = dashboard.read_bytes()
    presentation_before = {repo / path: (repo / path).read_bytes() for path in PRESENTATION_PATHS}
    counts: dict[str, int] = {}

    def fake_run(*arguments: str) -> None:
        script = arguments[0]
        counts[script] = counts.get(script, 0) + 1
        config = Path(arguments[arguments.index("--config") + 1])
        paths = yaml.safe_load(config.read_text())["paths"]
        staged_results = (config.parent / paths["results_dir"]).resolve()
        staged_results.mkdir(parents=True, exist_ok=True)
        if script == "scripts/write_run_summary.py":
            _fake_run_summary(staged_results)
        if script == "scripts/build_dashboard_data.py":
            staged_dashboard = config.parent / "dashboard/src/generated/release.json"
            staged_dashboard.parent.mkdir(parents=True, exist_ok=True)
            staged_dashboard.write_text("new dashboard\n")
        if script == failing_script and counts[script] == occurrence:
            raise subprocess.CalledProcessError(1, script)

    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(module, "_compile_report", _fake_compile_report)

    with pytest.raises(subprocess.CalledProcessError):
        module._run_release(repo / "config.yaml", refresh_data=False)

    assert _tree_digest(repo / "results") == results_before
    assert dashboard.read_bytes() == dashboard_before
    assert {path: path.read_bytes() for path in presentation_before} == presentation_before
    assert not list(repo.glob(".results-staging-*"))


def _manifest_fixture(tmp_path: Path):
    repo = _repo(tmp_path)
    (repo / "dashboard/src/generated/release.json").unlink()
    _git(repo, "add", "-u", "dashboard/src/generated/release.json")
    _git(repo, "commit", "-qm", "remove dashboard fixture")
    results = repo / "results"
    (results / "metrics").mkdir(exist_ok=True)
    (results / "metrics/validation.csv").write_text("metric,value\nMAE,1\n")
    (results / "runtime.tmp").write_text("not canonical\n")
    (results / "__pycache__").mkdir()
    (results / "__pycache__/cache.pyc").write_bytes(b"cache")
    (repo / "report").mkdir(exist_ok=True)
    (repo / "report/generated_results.tex").write_text("generated\n")
    _git(repo, "add", "report/generated_results.tex")
    _git(repo, "commit", "-qm", "report input")

    source_revision = _git(repo, "rev-parse", "HEAD")
    config_sha256 = sha256_file(repo / "config.yaml")
    record = {
        "schema_version": 1,
        "source_revision": source_revision,
        "config_sha256": config_sha256,
        "stream_id": "daily/test",
        "model_identity": "test",
        "ordered_feature_columns": [],
        "model_parameters": {},
        "seed": 42,
        "splits": {
            "train": ["2020-01-01", "2020-12-31"],
            "validation": ["2021-01-01", "2021-12-31"],
            "calibration": ["2022-01-01", "2022-12-31"],
            "retrospective_final": ["2023-01-01", "2023-12-31"],
        },
        "weather_strategy": "available_day_ahead",
        "point_state_policy": {},
        "interval_state_policy": {},
        "final_role": "retrospective_final",
        "validation_selection_evidence": "validation metrics",
        "rationale": "fixture",
    }
    records = {"daily/test": record}
    fingerprints = {"daily/test": protocol_fingerprint(record)}
    summary = _release_identity(
        repo / "config.yaml",
        results,
        fingerprints,
        artifact_root=repo,
    )
    return repo, results, records, summary


def test_release_identity_declares_only_canonical_outputs(tmp_path):
    repo, results, records, summary = _manifest_fixture(tmp_path)

    assert summary["source_revision"] == _git(repo, "rev-parse", "HEAD")
    assert summary["source_clean"] is True
    assert summary["config_sha256"] == sha256_file(repo / "config.yaml")
    assert summary["lock_sha256"] == sha256_file(repo / "uv.lock")
    assert len(summary["bundle_fingerprint"]) == 64
    assert summary["declared_artifacts"] == {
        "dashboard/src/generated/release.json": None,
        "report/generated_results.tex": sha256_file(repo / "report/generated_results.tex"),
        "report/technical-report-en.pdf": sha256_file(repo / "report/technical-report-en.pdf"),
        "results/metrics/validation.csv": sha256_file(results / "metrics/validation.csv"),
    }
    assert summary["artifact_hashes"] == {
        "metrics/validation.csv": sha256_file(results / "metrics/validation.csv")
    }
    _validate_release_manifest(summary, results, records, repo, repo / "config.yaml")

    staging = repo / ".results-staging-test"
    (staging / "results/metrics").mkdir(parents=True)
    (staging / "config.yaml").write_bytes((repo / "config.yaml").read_bytes())
    (staging / "results/metrics/validation.csv").write_text("metric,value\nMAE,1\n")
    (staging / "report").mkdir()
    (staging / "report/generated_results.tex").write_text("generated\n")
    (staging / "report/technical-report-en.pdf").write_bytes(b"staged pdf\n")
    staged = _release_identity(
        staging / "config.yaml",
        staging / "results",
        summary["protocol_fingerprints"],
    )
    assert staged["source_clean"] is True
    assert "metrics/validation.csv" in staged["artifact_hashes"]


def test_live_release_validation_allows_only_generated_output_changes(tmp_path):
    repo, results, records, _ = _manifest_fixture(tmp_path)
    generated = (*PRESENTATION_PATHS, Path("dashboard/src/generated/release.json"))
    for relative in generated:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("regenerated\n")

    run_pipeline._assert_clean_source(repo)
    refreshed = _release_identity(
        repo / "config.yaml",
        results,
        {stream_id: protocol_fingerprint(record) for stream_id, record in records.items()},
        artifact_root=repo,
    )
    assert refreshed["source_clean"] is True
    assert _source_is_clean(repo, repo)

    (repo / "source.py").write_text("changed = True\n")
    assert not _source_is_clean(repo, repo)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "changed",
        "missing_pdf",
        "changed_pdf",
        "extra_declared",
        "source",
        "config",
        "lock",
        "protocol",
        "bundle",
    ],
)
def test_release_manifest_rejects_hash_and_identity_conflicts(tmp_path, mutation):
    repo, results, records, summary = _manifest_fixture(tmp_path)
    summary = {**summary, "declared_artifacts": dict(summary["declared_artifacts"])}
    if mutation == "missing":
        (results / "metrics/validation.csv").unlink()
    elif mutation == "changed":
        (results / "metrics/validation.csv").write_text("changed\n")
    elif mutation == "missing_pdf":
        (repo / "report/technical-report-en.pdf").unlink()
    elif mutation == "changed_pdf":
        (repo / "report/technical-report-en.pdf").write_bytes(b"changed pdf\n")
    elif mutation == "extra_declared":
        summary["declared_artifacts"]["results/extra.csv"] = "0" * 64
    elif mutation == "source":
        summary["source_revision"] = "different"
    elif mutation == "config":
        summary["config_sha256"] = "0" * 64
    elif mutation == "lock":
        summary["lock_sha256"] = "0" * 64
    elif mutation == "protocol":
        summary["protocol_fingerprints"] = {"daily/test": "different"}
    else:
        summary["bundle_fingerprint"] = "0" * 64

    with pytest.raises((AssertionError, ValueError)):
        _validate_release_manifest(summary, results, records, repo, repo / "config.yaml")


def test_every_release_stage_script_imports_as_a_subprocess():
    """The pipeline runs each stage as a subprocess, not as a pytest import.

    A stage that imports a sibling script resolves only under pytest, where the
    repository root is already on sys.path. This runs the real invocation form so
    an ImportError surfaces here instead of mid-release.
    """
    root = Path(__file__).resolve().parents[1]
    stages = [
        *run_pipeline._RELEASE_STAGES,
        "scripts/validate_results.py",
        "scripts/build_dashboard_data.py",
    ]
    for stage in stages:
        result = subprocess.run(
            [sys.executable, stage, "--help"],
            cwd=root,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"{stage} failed to start:\n{result.stderr}"


def test_report_pdf_is_built_declared_validated_and_promoted_in_staging(
    tmp_path, monkeypatch
):
    module = _module()
    repo = _repo(tmp_path)
    monkeypatch.setattr(module, "ROOT", repo)
    events: list[str] = []
    fingerprint = "a" * 64
    summary_writes = 0

    def fake_run(*arguments: str) -> None:
        nonlocal summary_writes
        script = arguments[0]
        events.append(script)
        config = Path(arguments[arguments.index("--config") + 1])
        staged_results = config.parent / "results"
        staged_results.mkdir(parents=True, exist_ok=True)
        if script == "scripts/write_run_summary.py":
            summary_writes += 1
            (staged_results / "run_summary.json").write_text(
                json.dumps({"bundle_fingerprint": fingerprint})
            )
            if summary_writes == 2:
                assert (config.parent / "report/technical-report-en.pdf").read_bytes() == b"staged-pdf"
        elif script == "scripts/render_report_data.py":
            assert json.loads((staged_results / "run_summary.json").read_text())[
                "bundle_fingerprint"
            ] == fingerprint
            source = config.parent / "report/technical-report-en.tex"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text("staged report source\n")
            for relative in module._RELEASE_PRESENTATION_PATHS:
                path = config.parent / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"staged {relative.as_posix()}\n")
        elif script == "scripts/build_dashboard_data.py":
            dashboard = config.parent / "dashboard/src/generated/release.json"
            dashboard.parent.mkdir(parents=True, exist_ok=True)
            dashboard.write_text("staged dashboard\n")

    def fake_compile(staging_root: Path) -> None:
        events.append("tectonic")
        assert (staging_root / "report/technical-report-en.tex").is_file()
        (staging_root / "report/technical-report-en.pdf").write_bytes(b"staged-pdf")

    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(module, "_compile_report", fake_compile, raising=False)

    module._run_release(repo / "config.yaml", refresh_data=False)

    assert "report/technical-report-en.pdf" in PRESENTATION_ARTIFACTS
    assert events == [
        *module._RELEASE_STAGES[:-1],
        "tectonic",
        module._RELEASE_STAGES[-1],
        "scripts/validate_results.py",
        "scripts/build_dashboard_data.py",
        "scripts/validate_results.py",
    ]
    assert events.index("scripts/write_run_summary.py") < events.index(
        "scripts/render_report_data.py"
    )
    assert events.index("scripts/render_report_data.py") < events.index("tectonic")
    write_indexes = [
        index for index, event in enumerate(events) if event == "scripts/write_run_summary.py"
    ]
    assert write_indexes[0] < events.index("scripts/render_report_data.py")
    assert events.index("tectonic") < write_indexes[1]
    assert summary_writes == 2
    assert (repo / "report/technical-report-en.pdf").read_bytes() == b"staged-pdf"


def test_final_manifest_must_preserve_preliminary_bundle_fingerprint(tmp_path, monkeypatch):
    module = _module()
    repo = _repo(tmp_path)
    monkeypatch.setattr(module, "ROOT", repo)
    summary_writes = 0

    def fake_run(*arguments: str) -> None:
        nonlocal summary_writes
        script = arguments[0]
        config = Path(arguments[arguments.index("--config") + 1])
        results = config.parent / "results"
        results.mkdir(parents=True, exist_ok=True)
        if script == "scripts/write_run_summary.py":
            summary_writes += 1
            fingerprint = ("a" if summary_writes == 1 else "b") * 64
            (results / "run_summary.json").write_text(
                json.dumps({"bundle_fingerprint": fingerprint})
            )
        elif script == "scripts/render_report_data.py":
            for relative in module._RELEASE_PRESENTATION_PATHS:
                path = config.parent / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("staged\n")

    monkeypatch.setattr(module, "_run", fake_run)
    monkeypatch.setattr(module, "_compile_report", _fake_compile_report)

    before = _tree_digest(repo / "results")
    with pytest.raises(RuntimeError, match="bundle fingerprint changed"):
        module._run_release(repo / "config.yaml", refresh_data=False)

    assert summary_writes == 2
    assert _tree_digest(repo / "results") == before
    assert not list(repo.glob(".results-staging-*"))


def test_post_report_release_is_valid_and_rerunnable(tmp_path):
    repo, results, records, _ = _manifest_fixture(tmp_path)
    pdf = repo / "report/technical-report-en.pdf"
    pdf.write_bytes(b"regenerated-pdf")
    fingerprints = {
        stream_id: protocol_fingerprint(record) for stream_id, record in records.items()
    }

    run_pipeline._assert_clean_source(repo)
    summary = _release_identity(
        repo / "config.yaml",
        results,
        fingerprints,
        artifact_root=repo,
    )
    _validate_release_manifest(summary, results, records, repo, repo / "config.yaml")
    run_pipeline._assert_clean_source(repo)

    assert summary["declared_artifacts"]["report/technical-report-en.pdf"] == sha256_file(pdf)
