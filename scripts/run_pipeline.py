"""Generate the complete evaluated release from cached or refreshed data."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from loadfc.presentation import MANAGED_PRESENTATION_PATHS
from loadfc.tracking import PRESENTATION_ARTIFACTS, source_is_clean

ROOT = Path(__file__).resolve().parents[1]
_RELEASE_PRESENTATION_PATHS = (
    *MANAGED_PRESENTATION_PATHS,
    Path("report/generated_results.tex"),
    Path("report/technical-report-en.pdf"),
)
_RELEASE_STAGES = [
    "scripts/run_fetch.py",
    "scripts/run_features.py",
    "scripts/run_evaluate.py",
    "scripts/run_intervals.py",
    "scripts/run_hourly.py",
    "scripts/run_comparison.py",
    "scripts/run_error_analysis.py",
    "scripts/run_analysis.py",
    "scripts/run_shap.py",
    "scripts/write_run_summary.py",
    "scripts/render_report_data.py",
    "scripts/write_run_summary.py",
]


def _run(*arguments: str) -> None:
    command = [sys.executable, *arguments]
    print("\n+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _assert_clean_source(root: Path) -> None:
    if not source_is_clean(root, allowed_paths=(*_RELEASE_PRESENTATION_PATHS, *PRESENTATION_ARTIFACTS)):
        raise RuntimeError("canonical release requires a clean committed source")


def _compile_report(staging_root: Path) -> None:
    tectonic = shutil.which("tectonic")
    if tectonic is None:
        raise RuntimeError("Tectonic is required to build the canonical report PDF")
    command = [tectonic, "-X", "compile", "report/technical-report-en.tex"]
    print("\n+", " ".join(command), flush=True)
    subprocess.run(
        command,
        cwd=staging_root,
        check=True,
        env=os.environ | {"SOURCE_DATE_EPOCH": "0"},
    )


def _prepare_staging(config_path: Path) -> tuple[Path, Path, Path, Path]:
    config_path = config_path.resolve()
    raw = yaml.safe_load(config_path.read_text())
    paths = raw.get("paths")
    if not isinstance(paths, dict) or "results_dir" not in paths:
        raise ValueError("release config must declare paths.results_dir")
    results_path = Path(paths["results_dir"])
    if results_path.is_absolute():
        raise ValueError("release config paths.results_dir must be relative")
    canonical = (config_path.parent / results_path).resolve()
    canonical.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{canonical.name}-staging-", dir=canonical.parent)
    )
    staged_config = staging_root / "config.yaml"
    staged_config.write_bytes(config_path.read_bytes())
    for key, value in paths.items():
        relative = Path(value)
        if key == "results_dir" or relative.is_absolute():
            continue
        destination = (staging_root / relative).resolve()
        if not destination.is_relative_to(staging_root):
            raise ValueError(f"release config path escapes staging root: {key}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to((config_path.parent / relative).resolve(), target_is_directory=True)
    for relative in (Path("docs/EVALUATION_LEDGER.md"), *_RELEASE_PRESENTATION_PATHS):
        source = config_path.parent / relative
        if source.is_file():
            destination = staging_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    return staging_root, staged_config, (staging_root / results_path).resolve(), canonical


def _promote_release(
    staged_results: Path,
    canonical_results: Path,
    staged_dashboard: Path,
    canonical_dashboard: Path,
    presentation_files: tuple[tuple[Path, Path], ...],
) -> None:
    files = (
        (staged_results, canonical_results),
        (staged_dashboard, canonical_dashboard),
        *presentation_files,
    )
    backups = tuple(
        (canonical, canonical.with_name(f".{canonical.name}-backup"))
        for _, canonical in files
    )
    if any(backup.exists() for _, backup in backups):
        raise RuntimeError("stale release backup blocks coordinated promotion")
    if not staged_results.is_dir() or not all(path.is_file() for path, _ in files[1:]):
        raise RuntimeError("coordinated promotion requires every staged release artifact")
    for _, canonical in files:
        canonical.parent.mkdir(parents=True, exist_ok=True)
    backed_up: list[tuple[Path, Path]] = []
    promoted: list[tuple[Path, Path]] = []
    try:
        for canonical, backup in backups:
            if canonical.exists():
                canonical.replace(backup)
                backed_up.append((canonical, backup))
        for staged, canonical in files:
            staged.replace(canonical)
            promoted.append((staged, canonical))
    except BaseException:
        for staged, canonical in reversed(promoted):
            canonical.replace(staged)
        for canonical, backup in reversed(backed_up):
            backup.replace(canonical)
        raise
    for _, backup in backups:
        if backup.is_dir():
            shutil.rmtree(backup)
        elif backup.exists():
            backup.unlink()


def _run_release(config_path: Path, *, refresh_data: bool, keep_staging: bool = False) -> None:
    _assert_clean_source(ROOT)
    staging_root, staged_config, staged_results, canonical = _prepare_staging(config_path)
    staged_dashboard = staging_root / "dashboard/src/generated/release.json"
    canonical_dashboard = config_path.parent / "dashboard/src/generated/release.json"
    presentation_files = tuple(
        (staging_root / relative, config_path.parent / relative)
        for relative in _RELEASE_PRESENTATION_PATHS
    )
    config_args = ["--config", str(staged_config)]
    preliminary_fingerprint = None
    try:
        for script in _RELEASE_STAGES:
            extra = ["--refresh"] if script == "scripts/run_fetch.py" and refresh_data else []
            _run(script, *config_args, *extra)
            if script == "scripts/write_run_summary.py":
                fingerprint = json.loads((staged_results / "run_summary.json").read_text()).get(
                    "bundle_fingerprint"
                )
                if preliminary_fingerprint is None:
                    preliminary_fingerprint = fingerprint
                elif fingerprint != preliminary_fingerprint:
                    raise RuntimeError("bundle fingerprint changed during manifest finalization")
            elif script == "scripts/render_report_data.py":
                _compile_report(staging_root)
        _run("scripts/validate_results.py", *config_args)
        _run("scripts/build_dashboard_data.py", *config_args)
        _run("scripts/validate_results.py", *config_args)
        _promote_release(
            staged_results,
            canonical,
            staged_dashboard,
            canonical_dashboard,
            presentation_files,
        )
    except BaseException:
        if keep_staging:
            # A late stage failing discards every earlier stage's work, which is
            # expensive to recompute. Keep the tree so it can be re-run in place.
            print(f"\nstaging preserved for inspection: {staging_root}", flush=True)
            raise
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    else:
        shutil.rmtree(staging_root, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="keep the staging tree when a stage fails, so it can be re-run in place",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    _run_release(
        config_path,
        refresh_data=args.refresh_data,
        keep_staging=args.keep_staging,
    )


if __name__ == "__main__":
    main()
