"""Snapshot the skills.sh all-time install leaderboard, our popularity signal.

The endpoint was found in the skills.sh frontend bundle; it is not a documented API, so the
snapshot is committed and everything downstream reads the snapshot, never the live site.
"""

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

URL_TEMPLATE = "https://skills.sh/api/skills/all-time/{page}"


def fetch_leaderboard(pages: int, http: httpx.Client | None = None) -> dict[str, Any]:
    http = http or httpx.Client(timeout=30)
    skills: list[dict[str, Any]] = []
    total = None
    for page in range(pages):
        resp = http.get(URL_TEMPLATE.format(page=page))
        resp.raise_for_status()
        data = resp.json()
        total = data["total"]
        skills.extend(data["skills"])
        if not data["hasMore"]:
            break
        time.sleep(1)  # be polite to an undocumented endpoint
    for rank, skill in enumerate(skills, start=1):
        skill["rank"] = rank
    return {
        "fetched_at": datetime.now(UTC).isoformat(),
        "url_template": URL_TEMPLATE,
        "pages": pages,
        "site_total": total,
        "skills": skills,
    }


def save_snapshot(snapshot: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=1) + "\n")


def load_snapshot(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text())
    return data
