from typing import Any

import numpy as np

from router.corpus.curate import (
    non_latin_ratio,
    rule_cap,
    rule_near_duplicate,
    rule_non_english,
    rule_same_name,
    rule_short_description,
)
from router.corpus.curate_llm import rule_indistinguishable


def skill(
    id: str, name: str = "", desc: str = "does a useful thing well", **kw: Any
) -> dict[str, Any]:
    return {
        "id": id,
        "name": name or id,
        "description": desc,
        "is_official": kw.get("official", False),
        "installs": kw.get("installs", 1),
    }


def test_short_description() -> None:
    kept, removed = rule_short_description([skill("a", desc="too short"), skill("b")], 5)
    assert [s["id"] for s in kept] == ["b"]
    assert removed[0]["rule"] == "short_description"


def test_non_english() -> None:
    assert non_latin_ratio("Create a PDF report") == 0.0
    assert non_latin_ratio("飞书日历管理") == 1.0
    assert non_latin_ratio("Café déjà vu — naïve") == 0.0
    kept, removed = rule_non_english(
        [skill("a", desc="管理飞书日历和会议 calendar"), skill("b")], 0.1
    )
    assert [s["id"] for s in kept] == ["b"]
    assert removed[0]["id"] == "a"


def test_same_name_prefers_official_then_description_then_installs() -> None:
    group = [
        skill("fork/pdf", "pdf", installs=900),
        skill("orig/pdf", "pdf", official=True, installs=10),
        skill("other/pdf", "pdf", desc="a much longer and more detailed description", installs=5),
    ]
    kept, removed = rule_same_name(group)
    assert [s["id"] for s in kept] == ["orig/pdf"]
    assert {r["kept_id"] for r in removed} == {"orig/pdf"}

    kept, _ = rule_same_name(group[0:1] + group[2:])
    assert [s["id"] for s in kept] == ["other/pdf"]


def test_near_duplicate_keeps_preferred() -> None:
    a = skill("a", desc="short desc of a thing", installs=100)
    b = skill("b", desc="longer description of the same thing", installs=1)
    c = skill("c", desc="unrelated skill entirely here", installs=50)
    v = {
        "a": np.array([1.0, 0.0]),
        "b": np.array([0.99, 0.141]),  # cosine ~0.99 with a
        "c": np.array([0.0, 1.0]),
    }
    kept, removed = rule_near_duplicate([a, b, c], v, threshold=0.92)
    assert {s["id"] for s in kept} == {"b", "c"}
    assert removed == [
        {"id": "a", "rule": "near_duplicate", "installs": 100, "kept_id": "b", "cosine": 0.99}
    ]


def test_indistinguishable_drops_less_preferred_and_stops_chains() -> None:
    a = skill("a", desc="the longest description of all three", installs=1)
    b = skill("b", desc="medium description here", installs=1)
    c = skill("c", desc="short one here", installs=1)
    judgments = [
        {"a": "a", "b": "b", "cosine": 0.91, "verdict": "same", "reason": "r"},
        {"a": "b", "b": "c", "cosine": 0.90, "verdict": "same", "reason": "r"},
        {"a": "a", "b": "c", "cosine": 0.86, "verdict": "overlap", "reason": "r"},
    ]
    kept, removed = rule_indistinguishable([a, b, c], judgments)
    # b loses to a; the b~c pair is then skipped because b is already gone.
    assert [s["id"] for s in kept] == ["a", "c"]
    assert [(r["id"], r["kept_id"]) for r in removed] == [("b", "a")]


def test_cap_keeps_most_installed() -> None:
    skills = [skill(str(i), installs=i) for i in range(5)]
    kept, removed = rule_cap(skills, 3)
    assert [s["id"] for s in kept] == ["4", "3", "2"]
    assert [r["id"] for r in removed] == ["1", "0"]
