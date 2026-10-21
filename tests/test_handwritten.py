import random
from typing import Any

from router.evalset.handwritten import check_rows
from router.evalset.slots import draw_singles

EQ = {"o/w/writing-skills": "eq:a/s/skill-creator", "a/s/skill-creator": "eq:a/s/skill-creator"}


def skill(id: str, desc: str = "does a thing", domain: str = "d") -> dict[str, Any]:
    name = id.rsplit("/", 1)[1]
    return {
        "id": id,
        "name": name,
        "description_original": desc,
        "description": desc,
        "domain": domain,
    }


SKILLS = [
    skill("a/s/skill-creator", "create and test new agent skills"),
    skill("o/w/writing-skills", "write skills for agents with tests"),
    skill("a/s/pdf", "read and edit pdf files"),
]


def row(slot: str, prompt: str, gold: str) -> dict[str, str]:
    return {"slot": slot, "prompt": prompt, "gold": gold}


def test_check_rows_flags_non_canonical_gold_and_skips_blank_rows() -> None:
    rows = [
        row(
            "1",
            "I keep explaining the same workflow to my agent, make that reusable",
            "o/w/writing-skills",
        ),
        row("2", "", "a/s/pdf"),
    ]
    results = check_rows(rows, SKILLS, EQ)
    assert len(results) == 1
    assert results[0]["problems"] == ["not canonical: o/w/writing-skills (use a/s/skill-creator)"]


def test_check_rows_applies_filters_to_equivalent_members() -> None:
    # "writing-skills" is the name of a member of skill-creator's set: D2 must catch it.
    rows = [row("1", "help me with writing skills for my coding agent please", "a/s/skill-creator")]
    problems = check_rows(rows, SKILLS, EQ)[0]["problems"]
    assert any(p.startswith("D2") for p in problems)


def test_check_rows_enforces_cap() -> None:
    prompts = [
        "turn the scanned contract into something I can sign today",
        "the auditor sent forty pages and I need the totals from page nine",
        "split the handbook so each chapter is its own file for the wiki",
        "my landlord wants the lease back with my initials on every page",
    ]
    rows = [row(str(i), p, "a/s/pdf") for i, p in enumerate(prompts, start=1)]
    results = check_rows(rows, SKILLS, EQ)
    assert all(any(p.startswith("cap:") for p in r["problems"]) for r in results)


def test_draw_singles_is_proportional_and_seeded() -> None:
    eligible = [skill(f"r/x/s{i}", domain="big") for i in range(30)] + [
        skill(f"r/y/t{i}", domain="small") for i in range(10)
    ]
    first = draw_singles(eligible, 8, random.Random(1))
    again = draw_singles(eligible, 8, random.Random(1))
    assert [s["id"] for s in first] == [s["id"] for s in again]
    domains = sorted(s["domain"] for s in first)
    assert domains.count("big") == 6 and domains.count("small") == 2
