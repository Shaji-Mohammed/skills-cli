"""Results convention: every eval run gets its own folder under results/ with a manifest.

results/<UTC timestamp>_<name>/
    run.json           config, corpus/eval-set versions, git commit, dirty flag, versions
    metrics.json       written by the run
    predictions.jsonl  written by the run

Only runs with `git_dirty: false` may be quoted in the README or in phase summaries.
"""

import hashlib
import json
import platform
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from router.config import REPO_ROOT, RESULTS_DIR


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state(repo: Path = REPO_ROOT) -> tuple[str, bool]:
    """Return (commit sha, has uncommitted changes). 'unknown' if not a repo or no commits."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", True
    return sha, bool(status.strip())


def _package_versions() -> dict[str, str]:
    versions = {}
    for pkg in ("skill-router", "httpx", "typer"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "missing"
    return versions


@dataclass
class Run:
    path: Path

    def write_metrics(self, metrics: dict[str, Any]) -> None:
        (self.path / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    def write_predictions(self, rows: Iterable[dict[str, Any]]) -> None:
        with (self.path / "predictions.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")


def start_run(
    name: str,
    config: dict[str, Any],
    corpus_path: Path | None,
    evalset_path: Path | None,
    results_dir: Path = RESULTS_DIR,
    repo: Path = REPO_ROOT,
) -> Run:
    now = datetime.now(UTC)
    path = results_dir / f"{now.strftime('%Y%m%dT%H%M%SZ')}_{name}"
    path.mkdir(parents=True, exist_ok=False)
    commit, dirty = git_state(repo)
    manifest = {
        "name": name,
        "started_at": now.isoformat(),
        "config": config,
        "corpus_path": str(corpus_path) if corpus_path else None,
        "corpus_version": file_sha256(corpus_path) if corpus_path else None,
        "evalset_path": str(evalset_path) if evalset_path else None,
        "evalset_version": file_sha256(evalset_path) if evalset_path else None,
        "git_commit": commit,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "packages": _package_versions(),
    }
    (path / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return Run(path)
