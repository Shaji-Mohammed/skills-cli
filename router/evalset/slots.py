"""Target slots for the 50 hand-written implied prompts, chosen by a seeded draw.

A script picks the skills so the writer doesn't gravitate to familiar ones. Pairs come from
skill text only: one skill's SKILL.md or description naming another, then a qwen check that a
single realistic task needs both. No retriever is consulted.
"""

import csv
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from router.config import REPO_ROOT
from router.corpus.collect import CACHE_DIR
from router.corpus.curate_llm import _call, _system_user
from router.corpus.pipeline import read_body
from router.corpus.product_locked import flagged_ids, load_product_locked
from router.evalset.equivalence import EQUIVALENCE, SKILLS, load_equivalence
from router.evalset.filters import canonical_id
from router.llm import LLMClient
from router.runs import file_sha256, start_run

SHEET = REPO_ROOT / "evals" / "handwritten" / "implied_slots.csv"
SEED = 20261022
SINGLES, SINGLE_RESERVES = 35, 5
PAIRS, PAIR_RESERVES = 15, 5
MIN_CROSS_REPO_PAIRS = 5
MAX_PAIRS_PER_REPO = 2

COLUMNS = [
    "slot",
    "reserve",
    "swapped",  # yes: the writer replaced this slot; see swap_reason
    "swap_reason",
    "draw",  # initial | redraw1 | ...
    "kind",
    "source",
    "bucket",
    "gold",
    "domain",
    "skill_descriptions",
    "skill_md_paths",
    "pair_reason",
    "prompt",
    "helpful_excluded",
    "notes",
]

Skill = dict[str, Any]

COOCCUR_SYSTEM = """You judge whether two AI-agent skills are used together.

Answer true only if one realistic user task would REQUIRE both skills, each supplying a distinct
part of the work (for example, one produces data that the other turns into a deliverable).

Answer false if one is an alternative to the other, if one only refers the user to the other,
if one is a general version of the other, or if needing both in one task would be unusual.

Reply with only a JSON object, no other text:
{"cooccur": true | false, "reason": "<at most 20 words>"}"""


def cooccur_request(a: Skill, b: Skill) -> tuple[list[dict[str, str]], int]:
    user = (
        f"Skill A\nname: {a['name']}\ndescription: {a['description_original']}\n\n"
        f"Skill B\nname: {b['name']}\ndescription: {b['description_original']}"
    )
    return _system_user(COOCCUR_SYSTEM, user), 60


def _repo(skill_id: str) -> str:
    return skill_id.rsplit("/", 1)[0]


def _name(skill_id: str) -> str:
    return skill_id.rsplit("/", 1)[1]


def body_path(s: Skill) -> str:
    path = CACHE_DIR / s["source_repo"].replace("/", "__") / s["source_path"]
    return str(path.relative_to(REPO_ROOT))


def cross_reference_pairs(skills: list[Skill], eq: dict[str, str]) -> list[tuple[str, str]]:
    """Canonical pairs where one skill's description or SKILL.md names the other (multi-token
    names only, preferring a same-repo target when names collide). Umbrella pairs (one name
    is a prefix of the other) and pairs inside one equivalence set are excluded."""
    by_name: dict[str, list[Skill]] = {}
    for s in skills:
        by_name.setdefault(s["name"], []).append(s)
    pairs: set[tuple[str, str]] = set()
    for s in skills:
        text = s["description_original"] + "\n" + read_body(s)
        for name, targets in by_name.items():
            if "-" not in name or name == s["name"]:
                continue
            if not re.search(rf"(?<![\w/-]){re.escape(name)}(?![\w-])", text):
                continue
            same_repo = [t for t in targets if t["source_repo"] == s["source_repo"]]
            for t in same_repo or targets:
                a, b = sorted((canonical_id(s["id"], eq), canonical_id(t["id"], eq)))
                na, nb = _name(a), _name(b)
                if a != b and not (nb.startswith(na + "-") or na.startswith(nb + "-")):
                    pairs.add((a, b))
    return sorted(pairs)


def draw_pairs(
    client: LLMClient,
    by_id: dict[str, Skill],
    pool: list[tuple[str, str]],
    rng: random.Random,
    target: int,
    min_cross: int,
    used: set[str] | None = None,
    per_repo: Counter[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Walk the shuffled pool, cross-repo pairs first (until min_cross are accepted), judging
    each pair until `target` are accepted. `used` and `per_repo` carry the once-per-skill and
    per-repo limits over from slots already drawn. Returns (accepted, every judgment made)."""
    cross = [p for p in pool if _repo(p[0]) != _repo(p[1])]
    same = [p for p in pool if _repo(p[0]) == _repo(p[1])]
    rng.shuffle(cross)
    rng.shuffle(same)
    accepted: list[dict[str, Any]] = []
    judged: list[dict[str, Any]] = []
    used = set(used or ())
    per_repo = Counter(per_repo or {})

    def consider(candidates: list[tuple[str, str]], limit: int) -> None:
        for a, b in candidates:
            if len(accepted) >= limit:
                return
            repos = {_repo(a), _repo(b)}
            if a in used or b in used or any(per_repo[r] >= MAX_PAIRS_PER_REPO for r in repos):
                continue
            data = _call(client, cooccur_request(by_id[a], by_id[b])) or {}
            ok = data.get("cooccur") is True
            judged.append({"a": a, "b": b, "cooccur": ok, "reason": str(data.get("reason", ""))})
            if ok:
                accepted.append(judged[-1])
                used.update((a, b))
                per_repo.update(repos)

    consider(cross, min_cross)
    remaining = [p for p in cross if not any(p == (j["a"], j["b"]) for j in judged)] + same
    rng.shuffle(remaining)
    consider(remaining, target)
    return accepted, judged


def draw_singles(eligible: list[Skill], n: int, rng: random.Random) -> list[Skill]:
    """Proportional to each domain's share of the eligible skills (largest remainder)."""
    by_domain: dict[str, list[Skill]] = {}
    for s in sorted(eligible, key=lambda s: s["id"]):
        by_domain.setdefault(s["domain"], []).append(s)
    total = len(eligible)
    quotas = {d: n * len(v) / total for d, v in by_domain.items()}
    counts = {d: int(q) for d, q in quotas.items()}
    for d in sorted(quotas, key=lambda d: (-(quotas[d] - counts[d]), d))[
        : n - sum(counts.values())
    ]:
        counts[d] += 1
    picked = [s for d, v in sorted(by_domain.items()) for s in rng.sample(v, counts[d])]
    rng.shuffle(picked)
    return picked


def make_row(
    by_id: dict[str, Skill],
    slot: int,
    kind: str,
    ids: list[str],
    reserve: bool,
    draw: str,
    reason: str = "",
) -> dict[str, Any]:
    return {
        "slot": slot,
        "reserve": "yes" if reserve else "no",
        "swapped": "no",
        "swap_reason": "",
        "draw": draw,
        "kind": kind,
        "source": "hand_written",
        "bucket": "implied",
        "gold": ";".join(ids),
        "domain": ";".join(by_id[i]["domain"] for i in ids),
        "skill_descriptions": " || ".join(f"{i}: {by_id[i]['description_original']}" for i in ids),
        "skill_md_paths": ";".join(body_path(by_id[i]) for i in ids),
        "pair_reason": reason,
        "prompt": "",
        "helpful_excluded": "",
        "notes": "",
    }


def write_sheet(rows: list[dict[str, Any]]) -> None:
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    with SHEET.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def build(client: LLMClient, allow_dirty: bool = False) -> Path:
    skills = [json.loads(line) for line in SKILLS.read_text().splitlines() if line.strip()]
    eq = load_equivalence(EQUIVALENCE)
    by_id = {s["id"]: s for s in skills}
    eligible_ids = {canonical_id(s["id"], eq) for s in skills}
    run = start_run(
        "handwritten_slots",
        config={
            "seed": SEED,
            "singles": SINGLES,
            "single_reserves": SINGLE_RESERVES,
            "pairs": PAIRS,
            "pair_reserves": PAIR_RESERVES,
            "min_cross_repo_pairs": MIN_CROSS_REPO_PAIRS,
            "max_pairs_per_repo": MAX_PAIRS_PER_REPO,
            "pair_source": "cross-references in description_original or SKILL.md",
            "pair_check": "qwen cooccur prompt",
            "llm_model": client.model,
            "equivalence_sha256": file_sha256(EQUIVALENCE),
        },
        corpus_path=SKILLS,
        evalset_path=None,
        allow_dirty=allow_dirty,
    )
    rng = random.Random(SEED)
    pool = cross_reference_pairs(skills, eq)
    print(f"{len(pool)} cross-reference pairs; judging until {PAIRS + PAIR_RESERVES} co-occur")
    pairs, judged = draw_pairs(
        client, by_id, pool, rng, target=PAIRS + PAIR_RESERVES, min_cross=MIN_CROSS_REPO_PAIRS
    )
    used = {p["a"] for p in pairs} | {p["b"] for p in pairs}
    singles = draw_singles(
        [by_id[i] for i in sorted(eligible_ids - used)], SINGLES + SINGLE_RESERVES, rng
    )

    rows: list[dict[str, Any]] = []
    for n, s in enumerate(singles):
        rows.append(make_row(by_id, len(rows) + 1, "single", [s["id"]], n >= SINGLES, "initial"))
    for n, p in enumerate(pairs):
        rows.append(
            make_row(
                by_id, len(rows) + 1, "pair", [p["a"], p["b"]], n >= PAIRS, "initial", p["reason"]
            )
        )
    write_sheet(rows)
    (SHEET.parent / "pair_judgments.jsonl").write_text(
        "".join(json.dumps(j) + "\n" for j in judged)
    )

    main = [r for r in rows if r["reserve"] == "no"]
    run.write_metrics(
        {
            "cross_reference_pairs": len(pool),
            "pairs_judged": len(judged),
            "pairs_accepted": len(pairs),
            "cross_repo_pairs_accepted": sum(_repo(p["a"]) != _repo(p["b"]) for p in pairs),
            "slots": len(main),
            "reserves": len(rows) - len(main),
            "single_domains": dict(Counter(r["domain"] for r in main if r["kind"] == "single")),
            "repos_in_main_slots": len({_repo(i) for r in main for i in r["gold"].split(";")}),
            "sheet_sha256": file_sha256(SHEET),
        }
    )
    return run.path


# --- re-draws ------------------------------------------------------------------------------

SWAPS = REPO_ROOT / "evals" / "handwritten" / "swaps.json"
REDRAW_SEED = SEED + 1
# Products the writer named as product-locked (2026-10-22), matched as whole tokens in a
# skill's repo, name or original description. Used, with qwen's provisional flags, until the
# confirmed list in corpus/product_locked.json exists. Exclusion only: a false match costs a
# candidate, never adds one.
NAMED_PRODUCTS = {
    "runcomfy",
    "higgsfield",
    "tavily",
    "exa",
    "caveman",
    "agentix",
    "shoehorn",
    "kling",
    "firecrawl",
    "hyperframes",
}
NAMED_PRODUCT_SKILLS = {"stripe/ai/stripe-projects"}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def product_locked_exclusions(skills: list[Skill]) -> set[str]:
    confirmed = load_product_locked()
    if confirmed:
        return confirmed
    named = {
        s["id"]
        for s in skills
        if _tokens(f"{s['source_repo']} {s['name']} {s['description_original']}") & NAMED_PRODUCTS
    }
    return named | NAMED_PRODUCT_SKILLS | flagged_ids()


def redraw(
    client: LLMClient, singles: int, pairs: int, min_cross: int, allow_dirty: bool = False
) -> Path:
    """Mark the swaps in swaps.json and append replacement reserve rows, drawn with the same
    procedure: product-locked skills and every skill already in the sheet are excluded, and
    existing pair rows count toward the per-repo limit."""
    skills = [json.loads(line) for line in SKILLS.read_text().splitlines() if line.strip()]
    eq = load_equivalence(EQUIVALENCE)
    by_id = {s["id"]: s for s in skills}
    swaps: dict[str, str] = json.loads(SWAPS.read_text())
    with SHEET.open(newline="") as f:
        rows: list[dict[str, Any]] = list(csv.DictReader(f))
    earlier = {r.get("draw", "") for r in rows if r.get("draw", "").startswith("redraw")}
    draw = f"redraw{len(earlier) + 1}"
    for r in rows:
        r.setdefault("swapped", "no")
        r.setdefault("swap_reason", "")
        r.setdefault("draw", "initial")
        if r["slot"] in swaps:
            r["swapped"], r["swap_reason"] = "yes", swaps[r["slot"]]
    unknown = set(swaps) - {r["slot"] for r in rows}
    if unknown:
        raise ValueError(f"swaps.json names slots not in the sheet: {sorted(unknown)}")

    in_sheet = {g for r in rows for g in r["gold"].split(";")}
    locked = product_locked_exclusions(skills)
    excluded = in_sheet | locked
    live_pairs = [r for r in rows if r["kind"] == "pair" and r["swapped"] == "no"]
    per_repo = Counter(
        repo for r in live_pairs for repo in {_repo(g) for g in r["gold"].split(";")}
    )

    run = start_run(
        "handwritten_redraw",
        config={
            "seed": REDRAW_SEED,
            "draw": draw,
            "singles": singles,
            "pairs": pairs,
            "min_cross_repo_pairs": min_cross,
            "max_pairs_per_repo": MAX_PAIRS_PER_REPO,
            "swaps": swaps,
            "product_locked_source": "confirmed" if load_product_locked() else "named+qwen",
            "named_products": sorted(NAMED_PRODUCTS),
            "llm_model": client.model,
            "swaps_sha256": file_sha256(SWAPS),
        },
        corpus_path=SKILLS,
        evalset_path=None,
        allow_dirty=allow_dirty,
    )
    rng = random.Random(REDRAW_SEED)
    pool = [p for p in cross_reference_pairs(skills, eq) if not set(p) & excluded]
    accepted, judged = draw_pairs(
        client,
        by_id,
        pool,
        rng,
        target=pairs,
        min_cross=min_cross,
        used=in_sheet,
        per_repo=per_repo,
    )
    used = in_sheet | {p["a"] for p in accepted} | {p["b"] for p in accepted}
    eligible = sorted({canonical_id(s["id"], eq) for s in skills} - used - locked)
    new_singles = draw_singles([by_id[i] for i in eligible], singles, rng)

    next_slot = max(int(r["slot"]) for r in rows) + 1
    for s in new_singles:
        rows.append(make_row(by_id, next_slot, "single", [s["id"]], True, draw))
        next_slot += 1
    for p in accepted:
        rows.append(make_row(by_id, next_slot, "pair", [p["a"], p["b"]], True, draw, p["reason"]))
        next_slot += 1
    write_sheet(rows)
    with (SHEET.parent / "pair_judgments.jsonl").open("a") as f:
        f.write("".join(json.dumps({**j, "draw": draw}) + "\n" for j in judged))

    run.write_metrics(
        {
            "swapped_slots": sorted(swaps, key=int),
            "excluded_product_locked": len(locked),
            "pair_pool_after_exclusions": len(pool),
            "pairs_judged": len(judged),
            "pairs_accepted": len(accepted),
            "cross_repo_pairs_accepted": sum(_repo(p["a"]) != _repo(p["b"]) for p in accepted),
            "singles_drawn": len(new_singles),
            "sheet_sha256": file_sha256(SHEET),
        }
    )
    return run.path
