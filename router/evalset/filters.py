"""Gold-label canonicalization and the degenerate-prompt filters D1-D4 (evals/LABELING.md).

The same functions check hand-written and model-drafted prompts. None of them look at router
output.
"""

import re
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

MIN_WORDS = 6
MAX_WORDS = 120
NGRAM = 5
NEAR_DUPLICATE_COSINE = 0.95

Skill = dict[str, Any]


def canonical_id(skill_id: str, eq: Mapping[str, str]) -> str:
    """eq maps member id -> set id ('eq:<canonical id>'); skills in no set are their own."""
    set_id = eq.get(skill_id)
    return set_id.removeprefix("eq:") if set_id else skill_id


def canonical_gold(skill_ids: Iterable[str], eq: Mapping[str, str]) -> list[str]:
    """Gold labels always use canonical ids, deduplicated and sorted."""
    return sorted({canonical_id(s, eq) for s in skill_ids})


def members_of(canonical: str, eq: Mapping[str, str]) -> list[str]:
    """All skill ids interchangeable with a canonical id (itself included)."""
    set_id = f"eq:{canonical}"
    members = [m for m, s in eq.items() if s == set_id]
    return sorted(set(members) | {canonical})


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def d1_word_count(prompt: str) -> str | None:
    n = len(prompt.split())
    if n < MIN_WORDS or n > MAX_WORDS:
        return f"D1: {n} words (allowed {MIN_WORDS}-{MAX_WORDS})"
    return None


def d2_verbatim_name(prompt: str, gold_skills: Iterable[Skill]) -> str | None:
    """Multi-token skill names may not appear, joined by hyphens or by spaces
    ('grill-me' and 'grill me' both fail). Single-token names like 'pdf' are allowed."""
    text = " ".join(_words(prompt))
    for s in gold_skills:
        tokens = _words(s["name"])
        if len(tokens) >= 2 and re.search(rf"\b{' '.join(tokens)}\b", text):
            return f"D2: contains skill name '{s['name']}'"
    return None


def _ngrams(words: list[str]) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + NGRAM]) for i in range(len(words) - NGRAM + 1)}


def d3_description_overlap(prompt: str, gold_skills: Iterable[Skill]) -> str | None:
    """No 5-word sequence shared with the original or rewritten description of a gold skill."""
    grams = _ngrams(_words(prompt))
    for s in gold_skills:
        for field in ("description_original", "description"):
            shared = grams & _ngrams(_words(s[field]))
            if shared:
                return f"D3: shares '{' '.join(sorted(shared)[0])}' with {s['id']} {field}"
    return None


def d4_near_duplicates(vectors: np.ndarray) -> dict[int, tuple[int, float]]:
    """index -> (earlier index, cosine) for every prompt within NEAR_DUPLICATE_COSINE of an
    earlier one. Vectors must be L2-normalized; the later prompt is the one dropped."""
    sims = vectors @ vectors.T
    out: dict[int, tuple[int, float]] = {}
    for i in range(1, len(vectors)):
        j = int(np.argmax(sims[i, :i]))
        if sims[i, j] >= NEAR_DUPLICATE_COSINE:
            out[i] = (j, round(float(sims[i, j]), 4))
    return out


def check_prompt(prompt: str, gold_skills: list[Skill]) -> list[str]:
    """D1-D3 failures for one prompt. D4 needs the whole set; see d4_near_duplicates."""
    checks = (
        d1_word_count(prompt),
        d2_verbatim_name(prompt, gold_skills),
        d3_description_overlap(prompt, gold_skills),
    )
    return [c for c in checks if c is not None]
