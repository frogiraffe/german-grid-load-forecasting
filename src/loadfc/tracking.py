"""Release provenance and artifact hashing helpers."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

PRESENTATION_ARTIFACTS = (
    "report/generated_results.tex",
    "report/technical-report-en.pdf",
    "dashboard/src/generated/release.json",
)


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
