import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from router.evalset.equivalence import build_sets, candidate_pool, load_equivalence


def skill(id: str, repo: str, desc: str = "does things") -> dict[str, Any]:
    return {"id": f"{repo}/{id}", "name": id, "source_repo": repo, "description_original": desc}


def test_pool_finds_cross_vendor_umbrella_and_alias() -> None:
    skills = [
        skill("tdd", "a/x"),
        skill("test-driven-development", "b/y"),
        skill("firecrawl", "f/cli"),
        skill("firecrawl-scrape", "f/cli"),
        skill("pg", "p/s", "Deprecated compatibility name for pg-setup."),
        skill("pg-setup", "p/s"),
    ]
    v = np.eye(6, dtype=np.float32)
    v[1] = v[0]  # tdd ~ test-driven-development, cosine 1
    pool = {(c["a"], c["b"]): c["origin"] for c in candidate_pool(skills, v)}
    assert pool[("a/x/tdd", "b/y/test-driven-development")] == "cross_vendor"
    assert pool[("f/cli/firecrawl", "f/cli/firecrawl-scrape")] == "umbrella"
    assert pool[("p/s/pg", "p/s/pg-setup")] == "alias"


def row(n: int, a: str, b: str, decision: str, canonical: str = "") -> dict[str, str]:
    return {"row": str(n), "a": a, "b": b, "decision": decision, "canonical": canonical}


INSTALLS = {"tdd": 10, "test-driven-development": 50, "tdd-lite": 1, "pdf": 5, "xlsx": 5}


def test_build_sets_merges_transitively_and_picks_most_installed() -> None:
    rows = [
        row(1, "tdd", "test-driven-development", "accept"),
        row(2, "test-driven-development", "tdd-lite", "accept"),
        row(3, "pdf", "xlsx", "reject"),
    ]
    sets = build_sets(rows, INSTALLS)
    assert sets == [
        {
            "id": "eq:test-driven-development",
            "canonical": "test-driven-development",
            "members": ["tdd", "tdd-lite", "test-driven-development"],
        }
    ]


def test_build_sets_respects_reviewer_canonical_and_rejects_conflicts() -> None:
    rows = [row(1, "tdd", "test-driven-development", "accept", canonical="tdd")]
    assert build_sets(rows, INSTALLS)[0]["canonical"] == "tdd"
    conflict = [*rows, row(2, "tdd", "tdd-lite", "accept", canonical="tdd-lite")]
    with pytest.raises(ValueError, match="Conflicting"):
        build_sets(conflict, INSTALLS)


def test_build_sets_requires_a_decision_on_every_row() -> None:
    with pytest.raises(ValueError, match="accept/reject"):
        build_sets([row(1, "tdd", "pdf", "")], INSTALLS)


def test_load_equivalence_maps_members_to_set(tmp_path: Path) -> None:
    path = tmp_path / "eq.json"
    sets = [{"id": "eq:a", "canonical": "a", "members": ["a", "b"]}]
    path.write_text(json.dumps({"sets": sets}))
    assert load_equivalence(path) == {"a": "eq:a", "b": "eq:a"}
