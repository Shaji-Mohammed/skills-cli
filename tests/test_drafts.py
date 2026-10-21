from router.evalset.drafts import TARGETS, make_slot, trim

EQ = {
    "obra/superpowers/writing-skills": "eq:anthropics/skills/skill-creator",
    "anthropics/skills/skill-creator": "eq:anthropics/skills/skill-creator",
}


def test_draft_from_non_canonical_member_gets_canonical_gold() -> None:
    slot = make_slot(1, "explicit", "single", ["obra/superpowers/writing-skills"], EQ)
    assert slot["drafted_from"] == ["obra/superpowers/writing-skills"]
    assert slot["gold"] == ["anthropics/skills/skill-creator"]


def test_no_skill_slot_has_empty_gold() -> None:
    slot = make_slot(2, "no_skill", "none", [], EQ, topic="trivia")
    assert slot["gold"] == []
    assert slot["topic"] == "trivia"


def test_trim_keeps_targets_in_slot_order() -> None:
    drafts = [{"bucket": "no_skill", "kind": "none", "n": i} for i in range(90)]
    kept = trim(drafts)
    assert len(kept) == TARGETS[("no_skill", "none")]
    assert [d["n"] for d in kept] == list(range(TARGETS[("no_skill", "none")]))
