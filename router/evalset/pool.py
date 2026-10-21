"""Pool hand-written and drafted prompts, drop near-duplicates (D4), split dev/test, and build
the review sheets (evals/LABELING.md sections 5-7).

The test split is never printed: commands report counts and hashes only.
"""

import csv
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from router.config import REPO_ROOT
from router.embed import embed
from router.evalset.drafts import OUT as DRAFTS
from router.evalset.equivalence import EQUIVALENCE, SKILLS, load_equivalence
from router.evalset.filters import d4_near_duplicates
from router.evalset.handwritten import check_rows
from router.evalset.slots import SHEET
from router.runs import file_sha256, start_run

POOL = REPO_ROOT / "evals" / "pool.jsonl"
POOL_DROPPED = REPO_ROOT / "evals" / "pool_dropped.jsonl"
DEV_REVIEW = REPO_ROOT / "evals" / "review" / "dev_review.csv"
TEST_REVIEW = REPO_ROOT / "evals" / "review" / "test_review.csv"
SPLIT_SEED = 20261024
TEST_SIZE = 150
DEV_SAMPLE = 0.15
AID_K = 10

REVIEW_COLUMNS = [
    "id",
    "bucket",
    "source",
    "multi_skill",
    "prompt",
    "proposed_gold",
    "helpful_excluded",
    "aid_candidates",  # union of BM25 top-10 and embedding top-10, untuned
    "decision",  # accept | edit | drop
    "edited_gold",  # ;-separated canonical ids, when decision = edit
    "missing_skill",  # any REQUIRED skill not in proposed_gold (any corpus id)
    "notes",
]

Item = dict[str, Any]


def hand_written_items(skills: list[dict[str, Any]], eq: dict[str, str]) -> list[Item]:
    """Selected hand-written rows; every one must have a prompt and pass the checks."""
    with SHEET.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("use") == "yes"]
    blank = [r["slot"] for r in rows if not r["prompt"].strip()]
    if blank:
        raise ValueError(f"Hand-written slots without a prompt: {blank}")
    failing = [r for r in check_rows(rows, skills, eq) if r["problems"]]
    if failing:
        raise ValueError(f"Hand-written prompts fail checks: {[r['slot'] for r in failing]}")
    return [
        {
            "id": f"hw-{r['slot']}",
            "prompt": " ".join(r["prompt"].split()),
            "gold": sorted(g for g in r["gold"].split(";") if g),
            "bucket": "implied",
            "source": "hand_written",
            "helpful_excluded": r["helpful_excluded"],
        }
        for r in rows
    ]


def drafted_items() -> list[Item]:
    items = []
    for line in DRAFTS.read_text().splitlines():
        d = json.loads(line)
        items.append(
            {
                "id": f"md-{d['slot']}",
                "prompt": d["prompt"],
                "gold": d["gold"],
                "bucket": d["bucket"],
                "source": "model_drafted",
                "helpful_excluded": "",
            }
        )
    return items


def stratified_split(
    items: list[Item], test_size: int, rng: random.Random
) -> tuple[list[Item], list[Item]]:
    """Proportional allocation per bucket x source x multi-skill stratum (largest remainder),
    so the test set mirrors the pool exactly in each stratum's share."""
    strata: dict[tuple[str, str, bool], list[Item]] = {}
    for it in items:
        strata.setdefault((it["bucket"], it["source"], it["multi_skill"]), []).append(it)
    quotas = {k: test_size * len(v) / len(items) for k, v in strata.items()}
    counts = {k: int(q) for k, q in quotas.items()}
    for k in sorted(quotas, key=lambda k: (-(quotas[k] - counts[k]), k))[
        : test_size - sum(counts.values())
    ]:
        counts[k] += 1
    test_ids: set[str] = set()
    for k in sorted(strata):
        members = sorted(strata[k], key=lambda it: it["id"])
        test_ids.update(it["id"] for it in rng.sample(members, counts[k]))
    dev = [it for it in items if it["id"] not in test_ids]
    test = [it for it in items if it["id"] in test_ids]
    return dev, test


def review_aid(prompts: list[str], skills: list[dict[str, Any]]) -> list[list[str]]:
    """Union of BM25 top-k and embedding top-k over name + original description. Untuned and
    independent of any router config, so gold labels can't drift toward one retriever."""
    texts = [f"{s['name']} {s['description_original']}" for s in skills]
    bm25 = BM25Okapi([re.findall(r"[a-z0-9]+", t.lower()) for t in texts])
    skill_vecs = embed(texts)
    prompt_vecs = embed(prompts)
    out = []
    for prompt, pv in zip(prompts, prompt_vecs, strict=True):
        lexical = np.argsort(-bm25.get_scores(re.findall(r"[a-z0-9]+", prompt.lower())))[:AID_K]
        semantic = np.argsort(-(skill_vecs @ pv))[:AID_K]
        ids = dict.fromkeys(skills[int(i)]["id"] for i in [*semantic, *lexical])
        out.append(list(ids))
    return out


def write_review(path: Path, items: list[Item], aids: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for it, aid in zip(items, aids, strict=True):
            writer.writerow(
                {
                    "id": it["id"],
                    "bucket": it["bucket"],
                    "source": it["source"],
                    "multi_skill": "yes" if it["multi_skill"] else "no",
                    "prompt": it["prompt"],
                    "proposed_gold": ";".join(it["gold"]),
                    "helpful_excluded": it["helpful_excluded"],
                    "aid_candidates": ";".join(aid),
                    "decision": "",
                    "edited_gold": "",
                    "missing_skill": "",
                    "notes": "",
                }
            )


def build(allow_dirty: bool = False) -> Path:
    skills = [json.loads(line) for line in SKILLS.read_text().splitlines() if line.strip()]
    eq = load_equivalence(EQUIVALENCE)
    run = start_run(
        "pool_split",
        config={
            "split_seed": SPLIT_SEED,
            "test_size": TEST_SIZE,
            "dev_review_sample": DEV_SAMPLE,
            "aid_k": AID_K,
            "drafts_sha256": file_sha256(DRAFTS),
            "handwritten_sheet_sha256": file_sha256(SHEET),
        },
        corpus_path=SKILLS,
        evalset_path=None,
        allow_dirty=allow_dirty,
    )
    # Hand-written first, so D4 drops the later draft rather than a hand-written prompt.
    items = hand_written_items(skills, eq) + drafted_items()
    for it in items:
        it["multi_skill"] = len(it["gold"]) >= 2
    dupes = d4_near_duplicates(embed([it["prompt"] for it in items]))
    dropped = [
        {**items[i], "rule": "D4", "duplicate_of": items[j]["id"], "cosine": cos}
        for i, (j, cos) in dupes.items()
    ]
    items = [it for i, it in enumerate(items) if i not in dupes]

    POOL.write_text("".join(json.dumps(it, ensure_ascii=False) + "\n" for it in items))
    POOL_DROPPED.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in dropped))

    rng = random.Random(SPLIT_SEED)
    dev, test = stratified_split(items, TEST_SIZE, rng)
    multi = [it for it in dev if it["multi_skill"]]
    rest = sorted((it for it in dev if not it["multi_skill"]), key=lambda it: it["id"])
    sampled = rng.sample(rest, round(DEV_SAMPLE * len(rest)))
    dev_review = sorted(multi + sampled, key=lambda it: it["id"])
    for it in dev:
        it["review_status"] = "unreviewed"
    for it in dev_review:
        it["review_status"] = "sampled"
    for it in test:
        it["review_status"] = "full"

    write_review(DEV_REVIEW, dev_review, review_aid([it["prompt"] for it in dev_review], skills))
    write_review(TEST_REVIEW, test, review_aid([it["prompt"] for it in test], skills))
    # Unreviewed dev items are already final; keep them for apply_review.
    (REPO_ROOT / "evals" / "review" / "dev_unreviewed.jsonl").write_text(
        "".join(
            json.dumps(it, ensure_ascii=False) + "\n"
            for it in dev
            if it["review_status"] == "unreviewed"
        )
    )

    def counts(group: list[Item]) -> dict[str, int]:
        return dict(
            Counter(
                f"{it['bucket']}/{it['source']}/{'multi' if it['multi_skill'] else 'single'}"
                for it in group
            )
        )

    run.write_metrics(
        {
            "pooled_before_d4": len(items) + len(dropped),
            "dropped_d4": len(dropped),
            "pooled": len(items),
            "dev": len(dev),
            "test": len(test),
            "dev_review_rows": len(dev_review),
            "dev_by_stratum": counts(dev),
            "test_by_stratum": counts(test),
            "multi_skill_share": round(sum(it["multi_skill"] for it in items) / len(items), 4),
            "pool_sha256": file_sha256(POOL),
            "test_review_sha256": file_sha256(TEST_REVIEW),
        }
    )
    return run.path


DEV = REPO_ROOT / "evals" / "dev.jsonl"
TEST = REPO_ROOT / "evals" / "test.jsonl"
DEV_UNREVIEWED = REPO_ROOT / "evals" / "review" / "dev_unreviewed.jsonl"
DECISIONS = ("accept", "edit", "drop")


def reviewed_items(
    path: Path, status: str, known: set[str], eq: dict[str, str]
) -> tuple[list[Item], list[dict[str, str]]]:
    """Apply accept/edit/drop. missing_skill ids are added to the gold set either way."""
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    bad = [r["id"] for r in rows if r["decision"].strip() not in DECISIONS]
    if bad:
        raise ValueError(f"{path.name}: {len(bad)} rows without accept/edit/drop, e.g. {bad[:3]}")
    items, dropped = [], []
    for r in rows:
        decision = r["decision"].strip()
        if decision == "drop":
            dropped.append(r)
            continue
        base = r["edited_gold"] if decision == "edit" else r["proposed_gold"]
        gold = {g.strip() for g in f"{base};{r['missing_skill']}".split(";") if g.strip()}
        unknown = gold - known
        if unknown:
            raise ValueError(f"{path.name} {r['id']}: unknown skill ids {sorted(unknown)}")
        noncanon = [g for g in gold if eq.get(g, f"eq:{g}") != f"eq:{g}"]
        if noncanon:
            raise ValueError(f"{path.name} {r['id']}: use canonical ids, not {noncanon}")
        items.append(
            {
                "id": r["id"],
                "prompt": r["prompt"],
                "gold": sorted(gold),
                "bucket": r["bucket"],
                "source": r["source"],
                "multi_skill": len(gold) >= 2,
                "review_status": status,
                "review_decision": decision,
            }
        )
    return items, dropped


def apply_review(allow_dirty: bool = False) -> tuple[Path, str]:
    """Write evals/dev.jsonl and, once only, evals/test.jsonl. Returns (run path, test sha)."""
    if TEST.exists():
        raise FileExistsError("evals/test.jsonl already exists and is locked; refusing to rewrite")
    skills = [json.loads(line) for line in SKILLS.read_text().splitlines() if line.strip()]
    known = {s["id"] for s in skills}
    eq = load_equivalence(EQUIVALENCE)
    run = start_run(
        "apply_review",
        config={
            "dev_review_sha256": file_sha256(DEV_REVIEW),
            "test_review_sha256": file_sha256(TEST_REVIEW),
        },
        corpus_path=SKILLS,
        evalset_path=None,
        allow_dirty=allow_dirty,
    )
    test, test_dropped = reviewed_items(TEST_REVIEW, "full", known, eq)
    dev_reviewed, dev_dropped = reviewed_items(DEV_REVIEW, "sampled", known, eq)
    dev_unreviewed = [
        {**json.loads(line), "review_decision": ""}
        for line in DEV_UNREVIEWED.read_text().splitlines()
        if line.strip()
    ]
    dev = sorted(dev_reviewed + dev_unreviewed, key=lambda it: it["id"])
    test.sort(key=lambda it: it["id"])
    keep = ("id", "prompt", "gold", "bucket", "source", "multi_skill", "review_status")
    DEV.write_text(
        "".join(json.dumps({k: it[k] for k in keep}, ensure_ascii=False) + "\n" for it in dev)
    )
    TEST.write_text(
        "".join(json.dumps({k: it[k] for k in keep}, ensure_ascii=False) + "\n" for it in test)
    )

    def strata(group: list[Item]) -> dict[str, int]:
        return dict(Counter(it["bucket"] for it in group)) | {
            "multi_skill": sum(it["multi_skill"] for it in group),
            "hand_written": sum(it["source"] == "hand_written" for it in group),
        }

    sha = file_sha256(TEST)
    run.write_metrics(
        {
            "dev": len(dev),
            "dev_review_status": dict(Counter(it["review_status"] for it in dev)),
            "dev_review_decisions": dict(Counter(it["review_decision"] for it in dev_reviewed))
            | {"drop": len(dev_dropped)},
            "dev_by_bucket": strata(dev),
            "test": len(test),
            "test_review_decisions": dict(Counter(it["review_decision"] for it in test))
            | {"drop": len(test_dropped)},
            "test_by_bucket": strata(test),
            "dev_sha256": file_sha256(DEV),
            "test_sha256": sha,
        }
    )
    return run.path, sha
