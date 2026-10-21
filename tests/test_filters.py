from typing import Any

import numpy as np

from router.evalset.filters import (
    canonical_gold,
    canonical_id,
    check_prompt,
    d1_word_count,
    d2_verbatim_name,
    d3_description_overlap,
    d4_near_duplicates,
    members_of,
)

EQ = {
    "obra/superpowers/writing-skills": "eq:anthropics/skills/skill-creator",
    "anthropics/skills/skill-creator": "eq:anthropics/skills/skill-creator",
}


def skill(
    id: str, name: str, desc: str = "does a thing", rewritten: str | None = None
) -> dict[str, Any]:
    return {
        "id": id,
        "name": name,
        "description_original": desc,
        "description": rewritten or desc,
    }


# --- canonical gold ------------------------------------------------------------------------


def test_prompt_drafted_from_non_canonical_member_gets_canonical_gold() -> None:
    assert canonical_id("obra/superpowers/writing-skills", EQ) == "anthropics/skills/skill-creator"
    assert canonical_gold(["obra/superpowers/writing-skills"], EQ) == [
        "anthropics/skills/skill-creator"
    ]


def test_canonical_gold_dedupes_members_and_keeps_other_skills() -> None:
    gold = canonical_gold(
        [
            "anthropics/skills/pdf",
            "obra/superpowers/writing-skills",
            "anthropics/skills/skill-creator",
        ],
        EQ,
    )
    assert gold == ["anthropics/skills/pdf", "anthropics/skills/skill-creator"]


def test_members_of() -> None:
    assert members_of("anthropics/skills/skill-creator", EQ) == [
        "anthropics/skills/skill-creator",
        "obra/superpowers/writing-skills",
    ]
    assert members_of("anthropics/skills/pdf", EQ) == ["anthropics/skills/pdf"]


# --- D1-D4 ---------------------------------------------------------------------------------


def test_d1_word_count() -> None:
    assert d1_word_count("too short here") is not None
    assert d1_word_count("this prompt has exactly six words") is None
    assert d1_word_count(" ".join(["word"] * 121)) is not None


def test_d2_catches_hyphenated_and_spaced_names_but_allows_single_tokens() -> None:
    grill = [skill("m/grill-me", "grill-me")]
    assert d2_verbatim_name("Please grill-me on this plan", grill) is not None
    assert d2_verbatim_name("Please grill me on this plan", grill) is not None
    assert d2_verbatim_name("Ask me hard questions about my plan", grill) is None
    assert d2_verbatim_name("Merge these PDF files together", [skill("a/pdf", "pdf")]) is None


def test_d3_checks_both_description_versions() -> None:
    s = skill(
        "a/x",
        "x",
        desc="Use when the user wants to plan a product launch",
        rewritten="Builds a launch checklist with timelines and owners",
    )
    # "to plan a product launch" is a 5-word sequence in the original description.
    assert d3_description_overlap("I want to plan a product launch soon", [s]) is not None
    # Only 4 shared words in a row ("plan a product launch"): allowed.
    assert d3_description_overlap("help me plan a product launch next month", [s]) is None
    assert d3_description_overlap("I need a launch checklist with timelines and owners", [s])
    assert d3_description_overlap("we ship in May, what should we prepare first", [s]) is None


def test_d4_flags_the_later_near_duplicate() -> None:
    v = np.array([[1.0, 0.0], [0.0, 1.0], [0.999, 0.0447]], dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    dupes = d4_near_duplicates(v)
    assert list(dupes) == [2]
    assert dupes[2][0] == 0


def test_check_prompt_collects_all_failures() -> None:
    s = skill("m/grill-me", "grill-me", desc="a relentless interview to sharpen a plan")
    failures = check_prompt("grill me: a relentless interview to sharpen a plan", [s])
    assert [f[:2] for f in failures] == ["D2", "D3"]
