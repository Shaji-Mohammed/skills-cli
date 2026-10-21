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
    client: LLMClient, by_id: dict[str, Skill], pool: list[tuple[str, str]], rng: random.Random
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Walk the shuffled pool, cross-repo pairs first, judging each pair until enough are
    accepted. Returns (accepted, every judgment made)."""
    cross = [p for p in pool if _repo(p[0]) != _repo(p[1])]
    same = [p for p in pool if _repo(p[0]) == _repo(p[1])]
    rng.shuffle(cross)
    rng.shuffle(same)
    target = PAIRS + PAIR_RESERVES
    accepted: list[dict[str, Any]] = []
    judged: list[dict[str, Any]] = []
    used: set[str] = set()
    per_repo: Counter[str] = Counter()

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

    consider(cross, MIN_CROSS_REPO_PAIRS)
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
    pairs, judged = draw_pairs(client, by_id, pool, rng)
    used = {p["a"] for p in pairs} | {p["b"] for p in pairs}
    singles = draw_singles(
        [by_id[i] for i in sorted(eligible_ids - used)], SINGLES + SINGLE_RESERVES, rng
    )

    rows: list[dict[str, Any]] = []

    def add(kind: str, ids: list[str], reserve: bool, reason: str = "") -> None:
        rows.append(
            {
                "slot": len(rows) + 1,
                "reserve": "yes" if reserve else "no",
                "kind": kind,
                "source": "hand_written",
                "bucket": "implied",
                "gold": ";".join(ids),
                "domain": ";".join(by_id[i]["domain"] for i in ids),
                "skill_descriptions": " || ".join(
                    f"{i}: {by_id[i]['description_original']}" for i in ids
                ),
                "skill_md_paths": ";".join(body_path(by_id[i]) for i in ids),
                "pair_reason": reason,
                "prompt": "",
                "helpful_excluded": "",
                "notes": "",
            }
        )

    for n, s in enumerate(singles):
        add("single", [s["id"]], reserve=n >= SINGLES)
    for n, p in enumerate(pairs):
        add("pair", [p["a"], p["b"]], reserve=n >= PAIRS, reason=p["reason"])

    SHEET.parent.mkdir(parents=True, exist_ok=True)
    with SHEET.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
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
