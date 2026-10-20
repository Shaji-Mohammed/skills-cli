"""Collect skills listed in the leaderboard snapshot from their GitHub repos.

Each source repo is shallow-cloned into .cache/repos (git-ignored); the HEAD commit is recorded
per skill so the corpus can be rebuilt from exactly the same files.
"""

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from router.config import REPO_ROOT

CACHE_DIR = REPO_ROOT / ".cache" / "repos"
# skills.sh sources are GitHub "owner/repo" slugs, except a few hosted elsewhere (e.g. a domain).
GITHUB_SLUG = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.DOTALL)

# Checked in order against the start of a LICENSE file; first match wins.
LICENSE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Apache-2.0", re.compile(r"Apache License,?\s+Version 2\.0", re.I)),
    ("MIT", re.compile(r"\bMIT License\b|Permission is hereby granted, free of charge", re.I)),
    ("BSD-3-Clause", re.compile(r"Neither the name of", re.I)),
    ("BSD-2-Clause", re.compile(r"Redistribution and use in source and binary forms", re.I)),
    ("GPL-3.0", re.compile(r"GNU GENERAL PUBLIC LICENSE\s+Version 3", re.I)),
    ("AGPL-3.0", re.compile(r"GNU AFFERO GENERAL PUBLIC LICENSE", re.I)),
    ("MPL-2.0", re.compile(r"Mozilla Public License,?\s+(?:Version|v\.?)\s*2\.0", re.I)),
    ("CC-BY-4.0", re.compile(r"Creative Commons Attribution 4\.0", re.I)),
    ("Unlicense", re.compile(r"This is free and unencumbered software", re.I)),
]


@dataclass(frozen=True)
class SkillFile:
    path: Path
    frontmatter: dict[str, Any]
    body: str


def parse_skill_md(text: str) -> tuple[dict[str, Any], str]:
    """Split SKILL.md into (frontmatter, body). Tolerates YAML that strict parsing rejects."""
    m = FRONTMATTER.match(text)
    if not m:
        return {}, text
    raw, body = m.group(1), text[m.end() :]
    try:
        data = yaml.safe_load(raw)
        if isinstance(data, dict):
            return data, body
    except yaml.YAMLError:
        pass
    # Common failure: unquoted colons in the description. Fall back to top-level `key: value`.
    data = {}
    for line in raw.splitlines():
        km = re.match(r"^([A-Za-z_-]+):\s*(.*)$", line)
        if km:
            data[km.group(1)] = km.group(2).strip().strip("\"'")
    return data, body


def detect_license(repo_dir: Path) -> str:
    for f in sorted(repo_dir.iterdir()):
        if f.is_file() and f.name.upper().startswith(("LICENSE", "LICENCE", "COPYING")):
            head = f.read_text(errors="replace")[:3000]
            for spdx, pattern in LICENSE_PATTERNS:
                if pattern.search(head):
                    return spdx
            return "other"
    return "unknown"


def clone(slug: str, cache: Path = CACHE_DIR) -> tuple[Path, str]:
    """Shallow-clone (or reuse) a GitHub repo; return (dir, HEAD sha)."""
    dest = cache / slug.replace("/", "__")
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--quiet", f"https://github.com/{slug}.git", dest],
            check=True,
            capture_output=True,
            timeout=300,
            # Private or deleted repos would otherwise block on a credential prompt.
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=dest, capture_output=True, text=True, check=True
    ).stdout.strip()
    return dest, sha


def index_skill_files(repo_dir: Path) -> list[SkillFile]:
    files = []
    for path in sorted(repo_dir.rglob("*")):
        if path.name.lower() != "skill.md" or ".git" in path.parts or not path.is_file():
            continue
        fm, body = parse_skill_md(path.read_text(errors="replace"))
        files.append(SkillFile(path.relative_to(repo_dir), fm, body))
    return files


def match_skill(skill_id: str, files: list[SkillFile]) -> SkillFile | None:
    """Find the SKILL.md for a leaderboard skillId: by frontmatter name, else by folder name."""
    by_name = [f for f in files if str(f.frontmatter.get("name", "")).strip() == skill_id]
    by_dir = [f for f in files if f.path.parent.name == skill_id]
    for group in (by_name, by_dir):
        if group:
            # Several copies inside one repo (e.g. plugin mirrors): take the shallowest path.
            return min(group, key=lambda f: (len(f.path.parts), str(f.path)))
    return None


def collect(snapshot: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (collected skills, failures). Every leaderboard entry lands in exactly one list."""
    skills: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    repos: dict[str, tuple[Path, str, list[SkillFile], str] | str] = {}

    for entry in snapshot["skills"]:
        source, skill_id = entry["source"], entry["skillId"]
        base = {
            "source": source,
            "skill_id": skill_id,
            "rank": entry["rank"],
            "installs": entry["installs"],
        }
        if not GITHUB_SLUG.match(source):
            failures.append({**base, "reason": "not_github_source"})
            continue
        if source not in repos:
            try:
                repo_dir, sha = clone(source)
                repos[source] = (
                    repo_dir,
                    sha,
                    index_skill_files(repo_dir),
                    detect_license(repo_dir),
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                repos[source] = "clone_failed"
        repo = repos[source]
        if isinstance(repo, str):
            failures.append({**base, "reason": repo})
            continue
        _, sha, files, repo_license = repo
        match = match_skill(skill_id, files)
        if match is None:
            failures.append({**base, "reason": "skill_md_not_found"})
            continue
        fm = match.frontmatter
        description = fm.get("description")
        skills.append(
            {
                "id": f"{source}/{skill_id}",
                "name": skill_id,
                "description": " ".join(str(description).split()) if description else "",
                "source_repo": source,
                "source_path": str(match.path),
                "license": str(fm["license"]) if fm.get("license") else repo_license,
                "source_commit": sha,
                "installs": entry["installs"],
                "is_official": entry.get("isOfficial", False),
                "rank": entry["rank"],
            }
        )
    return skills, failures
