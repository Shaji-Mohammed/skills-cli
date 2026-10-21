"""Equivalence-set candidates, human review, and the frozen evals/equivalence.json.

Candidates come from skill text only (names and ORIGINAL descriptions), never from eval
prompts. See evals/LABELING.md section 2.
"""

import csv
import json
import random
import re
from collections import Counter
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np

from router.config import REPO_ROOT
from router.corpus.curate_llm import judge_pairs
from router.embed import DEFAULT_MODEL, embed
from router.llm import LLMClient
from router.runs import file_sha256, start_run
from router.stats import wilson

Skill = dict[str, Any]
Candidate = dict[str, Any]

SKILLS = REPO_ROOT / "corpus" / "skills.jsonl"
OVERLAPS = REPO_ROOT / "corpus" / "overlaps.jsonl"
REVIEW_CSV = REPO_ROOT / "evals" / "review" / "equivalence_candidates.csv"
POOL = REPO_ROOT / "evals" / "review" / "equivalence_pool.jsonl"
EQUIVALENCE = REPO_ROOT / "evals" / "equivalence.json"
LLM_CACHE = REPO_ROOT / "evals" / "raw" / "llm_cache.jsonl"

CROSS_VENDOR_K = 3  # nearest skills from other repos, per skill
CROSS_VENDOR_MIN_COSINE = 0.65
SAME_REPO_MIN_COSINE = 0.65
NAME_SIMILARITY = 0.6  # difflib ratio on names after dropping tokens common in the repo
REPO_COMMON_TOKEN_MIN = 3  # a name token in this many of a repo's skills is a vendor prefix
DISTINCT_SPOT_CHECK = 20
SEED = 20261021

ALIAS_TEXT = re.compile(r"deprecated|compatibility name|\balias\b|renamed|superseded", re.I)

REVIEW_COLUMNS = [
    "row",
    "origin",
    "a",
    "b",
    "cosine",
    "qwen_verdict",
    "qwen_reason",
    "a_description",
    "b_description",
    "decision",  # accept | reject
    "canonical",  # optional: a or b's id, to choose the label used in gold sets
    "note",
]


def skill_text(s: Skill) -> str:
    return f"{s['name']}: {s['description_original']}"


def _tokens(name: str) -> list[str]:
    return [t for t in re.split(r"[-_]", name) if t]


def candidate_pool(skills: list[Skill], vectors: np.ndarray) -> list[Candidate]:
    """All structural and similarity candidates, deduplicated, with their origin."""
    sims = vectors @ vectors.T
    n = len(skills)
    found: dict[tuple[int, int], str] = {}

    def add(i: int, j: int, origin: str) -> None:
        key = (min(i, j), max(i, j))
        # Structural origins (alias, umbrella) win over similarity ones for the same pair.
        if key not in found or origin in ("alias", "umbrella"):
            found[key] = origin

    for i in range(n):
        others = [
            j
            for j in np.argsort(-sims[i])
            if j != i
            and skills[j]["source_repo"] != skills[i]["source_repo"]
            and sims[i, j] >= CROSS_VENDOR_MIN_COSINE
        ]
        for j in others[:CROSS_VENDOR_K]:
            add(i, int(j), "cross_vendor")

    by_repo: dict[str, list[int]] = {}
    for i, s in enumerate(skills):
        by_repo.setdefault(s["source_repo"], []).append(i)
    for ids in by_repo.values():
        counts = Counter(t for i in ids for t in set(_tokens(skills[i]["name"])))
        common = {t for t, c in counts.items() if c >= REPO_COMMON_TOKEN_MIN}
        for x, a in enumerate(ids):
            for b in ids[x + 1 :]:
                na, nb = skills[a]["name"], skills[b]["name"]
                if nb.startswith(na + "-") or na.startswith(nb + "-"):
                    add(a, b, "umbrella")
                    continue
                ta = "-".join(t for t in _tokens(na) if t not in common)
                tb = "-".join(t for t in _tokens(nb) if t not in common)
                if (
                    ta
                    and tb
                    and SequenceMatcher(None, ta, tb).ratio() >= NAME_SIMILARITY
                    and sims[a, b] >= SAME_REPO_MIN_COSINE
                ):
                    add(a, b, "same_repo_name")

    index = {s["name"]: i for i, s in enumerate(skills)}
    for i, s in enumerate(skills):
        if not ALIAS_TEXT.search(s["description_original"]):
            continue
        for name, j in index.items():
            if (
                j != i
                and "-" in name
                and re.search(rf"\b{re.escape(name)}\b", s["description_original"])
            ):
                add(i, j, "alias")

    return [
        {
            "a": skills[i]["id"],
            "b": skills[j]["id"],
            "cosine": round(float(sims[i, j]), 4),
            "origin": origin,
        }
        for (i, j), origin in sorted(found.items())
    ]


def prescreen(client: LLMClient, skills: list[Skill], pool: list[Candidate]) -> list[Candidate]:
    """Run the curation pair judge on similarity candidates, using ORIGINAL descriptions."""
    by_id = {s["id"]: {**s, "description": s["description_original"]} for s in skills}
    targets = [c for c in pool if c["origin"] in ("cross_vendor", "same_repo_name")]
    verdicts = judge_pairs(client, [(c["cosine"], by_id[c["a"]], by_id[c["b"]]) for c in targets])
    judged = {(v["a"], v["b"]): v for v in verdicts}
    out = []
    for c in pool:
        v = judged.get((c["a"], c["b"]))
        out.append(
            {
                **c,
                "qwen_verdict": v["verdict"] if v else "",
                "qwen_reason": v["reason"] if v else "",
            }
        )
    return out


def review_rows(
    skills: list[Skill], screened: list[Candidate], overlaps: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Rows a person reviews: the curation overlaps, every alias and umbrella pair, every
    pair the pre-screen called same/overlap, and a random sample of its 'distinct' pairs."""
    desc = {s["id"]: s["description_original"] for s in skills}
    rows: list[dict[str, Any]] = [
        {
            **o,
            "origin": "curation_overlap",
            "qwen_verdict": o["verdict"],
            "qwen_reason": o["reason"],
        }
        for o in overlaps
    ]
    seen = {(r["a"], r["b"]) for r in rows} | {(r["b"], r["a"]) for r in rows}
    distinct = []
    for c in screened:
        if (c["a"], c["b"]) in seen:
            continue
        if c["origin"] in ("alias", "umbrella") or c["qwen_verdict"] in ("same", "overlap"):
            rows.append(c)
        elif c["qwen_verdict"] == "distinct":
            distinct.append(c)
    spot = random.Random(SEED).sample(distinct, min(DISTINCT_SPOT_CHECK, len(distinct)))
    rows += [{**c, "origin": f"{c['origin']}:distinct_spot_check"} for c in spot]

    order = ["curation_overlap", "alias", "umbrella", "same_repo_name", "cross_vendor"]
    rows.sort(
        key=lambda r: (order.index(r["origin"].split(":")[0]), ":" in r["origin"], -r["cosine"])
    )
    return [
        {
            "row": n,
            "origin": r["origin"],
            "a": r["a"],
            "b": r["b"],
            "cosine": r["cosine"],
            "qwen_verdict": r.get("qwen_verdict", ""),
            "qwen_reason": r.get("qwen_reason", ""),
            "a_description": desc[r["a"]],
            "b_description": desc[r["b"]],
            "decision": "",
            "canonical": "",
            "note": "",
        }
        for n, r in enumerate(rows, start=1)
    ]


def write_review_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


# --- freezing -----------------------------------------------------------------------------


def build_sets(rows: list[dict[str, str]], installs: dict[str, int]) -> list[dict[str, Any]]:
    """Merge accepted pairs into sets (transitively). Canonical member: the reviewer's choice
    if given, else the most-installed member. Conflicting reviewer choices are an error."""
    bad = [r["row"] for r in rows if r["decision"].strip() not in ("accept", "reject")]
    if bad:
        raise ValueError(f"Rows without accept/reject: {bad[:10]}{'...' if len(bad) > 10 else ''}")
    unknown = [s for r in rows for s in (r["a"], r["b"]) if s not in installs]
    if unknown:
        raise ValueError(f"Unknown skill ids: {unknown[:5]}")

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    accepted = [r for r in rows if r["decision"].strip() == "accept"]
    for r in accepted:
        parent[find(r["a"])] = find(r["b"])

    groups: dict[str, set[str]] = {}
    for x in list(parent):
        groups.setdefault(find(x), set()).add(x)

    chosen: dict[str, str] = {}
    for r in accepted:
        c = r["canonical"].strip()
        if not c:
            continue
        if c not in (r["a"], r["b"]):
            raise ValueError(f"Row {r['row']}: canonical must be a or b")
        root = find(c)
        if chosen.setdefault(root, c) != c:
            raise ValueError(f"Conflicting canonical choices in one set: {chosen[root]} vs {c}")

    sets = []
    for root, members in groups.items():
        canonical = chosen.get(root) or max(members, key=lambda m: (installs[m], m))
        sets.append({"id": f"eq:{canonical}", "canonical": canonical, "members": sorted(members)})
    return sorted(sets, key=lambda s: s["id"])


def generate(client: LLMClient, allow_dirty: bool = False) -> Path:
    """Build the pool, pre-screen it, write the review CSV and record a results run."""
    skills = [json.loads(line) for line in SKILLS.read_text().splitlines() if line.strip()]
    overlaps = [json.loads(line) for line in OVERLAPS.read_text().splitlines() if line.strip()]
    run = start_run(
        "equivalence_candidates",
        config={
            "cross_vendor_k": CROSS_VENDOR_K,
            "cross_vendor_min_cosine": CROSS_VENDOR_MIN_COSINE,
            "same_repo_min_cosine": SAME_REPO_MIN_COSINE,
            "name_similarity": NAME_SIMILARITY,
            "repo_common_token_min": REPO_COMMON_TOKEN_MIN,
            "distinct_spot_check": DISTINCT_SPOT_CHECK,
            "seed": SEED,
            "embedding_model": DEFAULT_MODEL,
            "embedding_text": "name: description_original",
            "llm_model": client.model,
        },
        corpus_path=SKILLS,
        evalset_path=None,
        allow_dirty=allow_dirty,
    )
    pool = candidate_pool(skills, embed([skill_text(s) for s in skills]))
    print(f"{len(pool)} candidate pairs; pre-screening the similarity ones with {client.model}")
    screened = prescreen(client, skills, pool)
    rows = review_rows(skills, screened, overlaps)

    POOL.parent.mkdir(parents=True, exist_ok=True)
    POOL.write_text("".join(json.dumps(c) + "\n" for c in screened))
    write_review_csv(REVIEW_CSV, rows)
    run.write_metrics(
        {
            "pool_by_origin": dict(Counter(c["origin"] for c in pool)),
            "prescreen_verdicts": dict(
                Counter(c["qwen_verdict"] or "not_screened" for c in screened)
            ),
            "review_rows": len(rows),
            "review_rows_by_origin": dict(Counter(r["origin"] for r in rows)),
        }
    )
    return run.path


def set_pair_audit(sets: list[dict[str, Any]], rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Every member pair inside a set that a reviewer didn't accept directly: either never
    reviewed (merged only through other accepts) or reviewed and rejected (a conflict)."""
    decided = {frozenset((r["a"], r["b"])): r for r in rows}
    out = []
    for s in sets:
        members = s["members"]
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                row = decided.get(frozenset((a, b)))
                if row is None:
                    out.append({"set": s["id"], "a": a, "b": b, "status": "never_reviewed"})
                elif row["decision"].strip() == "reject":
                    out.append(
                        {
                            "set": s["id"],
                            "a": a,
                            "b": b,
                            "status": "rejected_but_merged",
                            "row": row["row"],
                        }
                    )
    return out


def prescreen_agreement(rows: list[dict[str, str]]) -> dict[str, Any]:
    """qwen's verdict against the reviewer's decision, for rows qwen actually judged.
    Curation-overlap rows carry the curation judge's verdict; alias and umbrella rows were
    never screened."""
    table: dict[str, Counter[str]] = {}
    for r in rows:
        verdict = r["qwen_verdict"].strip() or "not_screened"
        if r["origin"].endswith(":distinct_spot_check"):
            verdict = "distinct (spot check)"
        table.setdefault(verdict, Counter())[r["decision"].strip()] += 1

    def accept_rate(verdict: str) -> dict[str, Any] | None:
        c = table.get(verdict)
        if not c:
            return None
        n = c["accept"] + c["reject"]
        lo, hi = wilson(c["accept"], n)
        return {
            "accept": c["accept"],
            "n": n,
            "rate": round(c["accept"] / n, 4),
            "wilson95": [round(lo, 4), round(hi, 4)],
        }

    return {
        "table": {v: dict(c) for v, c in sorted(table.items())},
        "accept_rate_by_verdict": {v: accept_rate(v) for v in sorted(table)},
    }


def freeze(allow_dirty: bool = False) -> tuple[dict[str, Any], Path]:
    """Turn the reviewed CSV into evals/equivalence.json and record the freeze in results/."""
    run = start_run(
        "equivalence_freeze",
        config={
            "review_csv": str(REVIEW_CSV.relative_to(REPO_ROOT)),
            "review_csv_sha256": file_sha256(REVIEW_CSV),
        },
        corpus_path=SKILLS,
        evalset_path=None,
        allow_dirty=allow_dirty,
    )
    with REVIEW_CSV.open(newline="") as f:
        rows = list(csv.DictReader(f))
    skills = [json.loads(line) for line in SKILLS.read_text().splitlines() if line.strip()]
    sets = build_sets(rows, {s["id"]: s["installs"] for s in skills})
    data = {
        "frozen_at": datetime.now(UTC).isoformat(),
        "review_csv_sha256": file_sha256(REVIEW_CSV),
        "corpus_version": file_sha256(SKILLS),
        "sets": sets,
    }
    EQUIVALENCE.write_text(json.dumps(data, indent=2) + "\n")
    audit = set_pair_audit(sets, rows)
    run.write_metrics(
        {
            "review_decisions": dict(Counter(r["decision"].strip() for r in rows)),
            "sets": len(sets),
            "skills_in_sets": sum(len(s["members"]) for s in sets),
            "set_sizes": dict(Counter(len(s["members"]) for s in sets)),
            "set_pair_audit": audit,
            "prescreen_agreement": prescreen_agreement(rows),
            "equivalence_sha256": file_sha256(EQUIVALENCE),
        }
    )
    return data, run.path


def load_equivalence(path: Path) -> dict[str, str]:
    """skill id -> set id, for router.score. Skills outside every set are absent."""
    data = json.loads(path.read_text())
    return {m: s["id"] for s in data["sets"] for m in s["members"]}
