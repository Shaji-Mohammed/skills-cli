"""Curation rules. Each rule takes the surviving skills and returns (kept, removed).

Rules run in a fixed order (see RULE_ORDER in run_curation) and every removal records the rule
and, for duplicates, which skill was kept instead, so CURATION.md can be generated from data.
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

Skill = dict[str, Any]
Removed = dict[str, Any]


@dataclass(frozen=True)
class CurationParams:
    min_description_words: int = 5
    max_non_latin_ratio: float = 0.10
    near_duplicate_cosine: float = 0.92
    cap: int = 500


def preference(skill: Skill) -> tuple[bool, int, int]:
    """Which copy to keep among duplicates: official source, then fuller description, then
    installs. Description length is our proxy for the roadmap's 'best-described copy'."""
    return (bool(skill["is_official"]), len(skill["description"]), skill["installs"])


def _removed(skill: Skill, rule: str, **extra: Any) -> Removed:
    return {"id": skill["id"], "rule": rule, "installs": skill["installs"], **extra}


def rule_short_description(
    skills: list[Skill], min_words: int
) -> tuple[list[Skill], list[Removed]]:
    kept, removed = [], []
    for s in skills:
        n = len(s["description"].split())
        if n < min_words:
            removed.append(_removed(s, "short_description", words=n))
        else:
            kept.append(s)
    return kept, removed


NON_LATIN = re.compile("[^\\x00-\\u024f\\u2000-\\u206f]")  # outside Latin scripts + punctuation


def non_latin_ratio(text: str) -> float:
    letters = [c for c in text if not c.isspace()]
    return len(NON_LATIN.findall(text)) / len(letters) if letters else 0.0


def rule_non_english(skills: list[Skill], max_ratio: float) -> tuple[list[Skill], list[Removed]]:
    # Eval prompts are English; a mostly-CJK description can't be matched fairly by any stage.
    kept, removed = [], []
    for s in skills:
        ratio = non_latin_ratio(s["description"])
        if ratio > max_ratio:
            removed.append(_removed(s, "non_english", non_latin_ratio=round(ratio, 3)))
        else:
            kept.append(s)
    return kept, removed


def rule_same_name(skills: list[Skill]) -> tuple[list[Skill], list[Removed]]:
    groups: dict[str, list[Skill]] = defaultdict(list)
    for s in skills:
        groups[s["name"]].append(s)
    kept, removed = [], []
    for group in groups.values():
        best = max(group, key=preference)
        kept.append(best)
        removed += [_removed(s, "same_name", kept_id=best["id"]) for s in group if s is not best]
    return kept, removed


def rule_near_duplicate(
    skills: list[Skill], vectors: dict[str, np.ndarray], threshold: float
) -> tuple[list[Skill], list[Removed]]:
    """Greedy: visit skills from most to least preferred; drop any whose description is within
    `threshold` cosine of an already-kept skill. Deterministic and order-independent of input."""
    kept: list[Skill] = []
    kept_vecs: list[np.ndarray] = []
    removed: list[Removed] = []
    for s in sorted(skills, key=preference, reverse=True):
        v = vectors[s["id"]]
        if kept_vecs:
            sims = np.stack(kept_vecs) @ v
            j = int(np.argmax(sims))
            if sims[j] >= threshold:
                removed.append(
                    _removed(
                        s, "near_duplicate", kept_id=kept[j]["id"], cosine=round(float(sims[j]), 4)
                    )
                )
                continue
        kept.append(s)
        kept_vecs.append(v)
    return kept, removed


def rule_cap(skills: list[Skill], cap: int) -> tuple[list[Skill], list[Removed]]:
    ranked = sorted(skills, key=lambda s: (-s["installs"], s["id"]))
    return ranked[:cap], [_removed(s, "below_cap") for s in ranked[cap:]]
