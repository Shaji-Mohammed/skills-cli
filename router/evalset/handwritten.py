"""Check hand-written prompts with the same filters as model drafts (D1-D4), plus label checks."""

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from router.embed import embed
from router.evalset.equivalence import EQUIVALENCE, SKILLS, load_equivalence
from router.evalset.filters import canonical_id, check_prompt, d4_near_duplicates, members_of

MAX_GOLD_SETS_PER_SKILL = 3


def read_sheet(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def check_rows(
    rows: list[dict[str, str]], skills: list[dict[str, Any]], eq: dict[str, str]
) -> list[dict[str, Any]]:
    """One result per filled row: its slot and every problem found (empty list = passes)."""
    by_id = {s["id"]: s for s in skills}
    filled = [r for r in rows if r["prompt"].strip()]
    results: list[dict[str, Any]] = []
    for r in filled:
        problems: list[str] = []
        gold = [g.strip() for g in r["gold"].split(";") if g.strip()]
        unknown = [g for g in gold if g not in by_id]
        problems += [f"unknown skill id: {g}" for g in unknown]
        problems += [
            f"not canonical: {g} (use {canonical_id(g, eq)})"
            for g in gold
            if g in by_id and canonical_id(g, eq) != g
        ]
        # D2/D3 check every member of each gold skill's equivalence set, so a prompt can't
        # leak a synonym's name or description either.
        gold_skills = [by_id[m] for g in gold if g in by_id for m in members_of(g, eq)]
        problems += check_prompt(r["prompt"], gold_skills)
        results.append({"slot": r["slot"], "problems": problems, "gold": gold})

    if filled:
        vectors = embed([r["prompt"] for r in filled])
        for i, (j, cos) in d4_near_duplicates(vectors).items():
            results[i]["problems"].append(f"D4: cosine {cos} with slot {filled[j]['slot']}")

    usage = Counter(g for res in results for g in res["gold"])
    for res in results:
        over = [g for g in res["gold"] if usage[g] > MAX_GOLD_SETS_PER_SKILL]
        res["problems"] += [f"cap: {g} is gold in {usage[g]} prompts" for g in over]
    return results


def check_sheet(path: Path) -> tuple[int, list[dict[str, Any]]]:
    rows = read_sheet(path)
    skills = [json.loads(line) for line in SKILLS.read_text().splitlines() if line.strip()]
    return len(rows), check_rows(rows, skills, load_equivalence(EQUIVALENCE))
