"""Run every curation rule in order, write the corpus, and record the run in results/."""

import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from router.config import REPO_ROOT
from router.corpus import curate, curate_llm
from router.corpus.collect import CACHE_DIR, parse_skill_md
from router.corpus.curate import CurationParams, Removed, Skill
from router.embed import DEFAULT_MODEL, embed
from router.llm import LLMClient
from router.runs import file_sha256, start_run

CORPUS_DIR = REPO_ROOT / "corpus"
SNAPSHOT = CORPUS_DIR / "sources" / "skills_sh_alltime.json"
RAW = CORPUS_DIR / "raw" / "skills_raw.jsonl"
RAW_FAILURES = CORPUS_DIR / "raw" / "collect_failures.jsonl"
SKILLS = CORPUS_DIR / "skills.jsonl"
REMOVED = CORPUS_DIR / "removed.jsonl"
OVERLAPS = CORPUS_DIR / "overlaps.jsonl"
PAIR_JUDGMENTS = CORPUS_DIR / "raw" / "pair_judgments.jsonl"
LLM_CACHE = CORPUS_DIR / "raw" / "llm_cache.jsonl"

# Pairs below the 0.92 auto-dedup threshold but at or above this go to the LLM pair judge.
# The run records nearest-neighbour cosine percentiles so this choice can be checked.
PAIR_JUDGE_MIN_COSINE = 0.85
MAX_DOMAIN_SHARE = 0.35

# Free-text license spellings seen in frontmatter, normalized so counts aren't split.
LICENSE_ALIASES = {"Apache 2.0": "Apache-2.0", "Apache License 2.0": "Apache-2.0"}

# Collection failures map onto curation rules. "skill_md_not_found" means the skill is on the
# leaderboard (installs are cumulative) but no longer exists in the repo's current HEAD.
FAILURE_RULES = {
    "not_github_source": "not_github_source",
    "clone_failed": "clone_failed",
    "skill_md_not_found": "not_in_current_repo",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_body(s: Skill) -> str:
    path = CACHE_DIR / s["source_repo"].replace("/", "__") / s["source_path"]
    return parse_skill_md(path.read_text(errors="replace"))[1]


def prefix_cluster_density(
    skills: list[Skill], vectors: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    """Mean pairwise description cosine per name-prefix cluster (3+ members). Diagnostic only."""
    groups: dict[str, list[str]] = defaultdict(list)
    for s in skills:
        groups[s["name"].split("-")[0]].append(s["id"])
    rows = []
    for prefix, ids in groups.items():
        if len(ids) < 3:
            continue
        m = np.stack([vectors[i] for i in ids])
        sims = (m @ m.T)[np.triu_indices(len(ids), 1)]
        rows.append(
            {
                "prefix": prefix,
                "size": len(ids),
                "mean_cosine": round(float(sims.mean()), 3),
                "max_cosine": round(float(sims.max()), 3),
            }
        )
    return sorted(rows, key=lambda r: -r["mean_cosine"])


def run_curation(client: LLMClient, params: CurationParams | None = None) -> Path:
    params = params or CurationParams()
    snapshot_size = len(json.loads(SNAPSHOT.read_text())["skills"])
    skills = read_jsonl(RAW)
    for s in skills:
        s["license"] = LICENSE_ALIASES.get(s["license"], s["license"])

    removed: list[Removed] = [
        {**f, "id": f"{f['source']}/{f['skill_id']}", "rule": FAILURE_RULES[f["reason"]]}
        for f in read_jsonl(RAW_FAILURES)
    ]
    steps: list[tuple[str, int, int]] = []  # (rule, removed, remaining)

    def step(rule: str, result: tuple[list[Skill], list[Removed]]) -> list[Skill]:
        kept, dropped = result
        removed.extend(dropped)
        steps.append((rule, len(dropped), len(kept)))
        return kept

    remaining = snapshot_size
    for rule in FAILURE_RULES.values():
        n = sum(1 for f in removed if f["rule"] == rule)
        remaining -= n
        steps.append((rule, n, remaining))
    assert remaining == len(skills), "every leaderboard entry is either collected or a failure"

    skills = step(
        "short_description", curate.rule_short_description(skills, params.min_description_words)
    )
    skills = step("non_english", curate.rule_non_english(skills, params.max_non_latin_ratio))
    skills = step("same_name", curate.rule_same_name(skills))

    matrix = embed([s["description"] for s in skills])
    vectors = {s["id"]: matrix[i] for i, s in enumerate(skills)}
    skills = step(
        "near_duplicate",
        curate.rule_near_duplicate(skills, vectors, params.near_duplicate_cosine),
    )

    m = np.stack([vectors[s["id"]] for s in skills])
    sims = m @ m.T
    np.fill_diagonal(sims, -1.0)
    nn_percentiles = {
        f"p{p}": round(float(v), 3)
        for p, v in zip(
            (50, 75, 90, 95, 99), np.percentile(sims.max(axis=1), [50, 75, 90, 95, 99]), strict=True
        )
    }

    pairs = curate_llm.candidate_pairs(
        skills, vectors, PAIR_JUDGE_MIN_COSINE, params.near_duplicate_cosine
    )
    print(f"Judging {len(pairs)} similar pairs with {client.model}")
    judgments = curate_llm.judge_pairs(client, pairs)
    write_jsonl(PAIR_JUDGMENTS, judgments)
    skills = step("indistinguishable", curate_llm.rule_indistinguishable(skills, judgments))
    skills = step("below_cap", curate.rule_cap(skills, params.cap))

    print(f"Classifying {len(skills)} skills")
    labels = curate_llm.classify(client, skills)
    needs_rewrite = [s for s in skills if not labels[s["id"]]["states_what"]]
    print(f"Rewriting {len(needs_rewrite)} trigger-only descriptions")
    rewrites = curate_llm.rewrite(client, needs_rewrite, read_body)
    for s in skills:
        s["domain"] = labels[s["id"]]["domain"]
        if (new := rewrites.get(s["id"])) is not None:
            s["description_original"] = s["description"]
            s["description"] = new

    kept_ids = {s["id"] for s in skills}
    overlaps = [j for j in judgments if j["verdict"] == "overlap" and {j["a"], j["b"]} <= kept_ids]

    skills.sort(key=lambda s: s["id"])
    write_jsonl(SKILLS, skills)
    write_jsonl(REMOVED, sorted(removed, key=lambda r: (r["rule"], r["id"])))
    write_jsonl(OVERLAPS, overlaps)

    domains = Counter(s["domain"] for s in skills)
    final_vectors = {s["id"]: vectors[s["id"]] for s in skills}
    metrics = {
        "leaderboard_entries": snapshot_size,
        "final_skills": len(skills),
        "steps": [{"rule": r, "removed": n, "remaining": k} for r, n, k in steps],
        "nearest_neighbour_cosine_after_near_duplicate": nn_percentiles,
        "pairs_judged": len(pairs),
        "pair_verdicts": dict(Counter(str(j["verdict"]) for j in judgments)),
        "overlap_pairs_kept": len(overlaps),
        "flagged_trigger_only": len(needs_rewrite),
        "rewritten_descriptions": sum(v is not None for v in rewrites.values()),
        "parse_failures": {
            "pairs": sum(j["verdict"] == "parse_failed" for j in judgments),
            "classify": sum(v["domain"] == "parse_failed" for v in labels.values()),
            "rewrite": sum(v is None for v in rewrites.values()),
        },
        "domains": dict(domains.most_common()),
        "max_domain_share": round(max(domains.values()) / len(skills), 4),
        "max_domain_share_ok": max(domains.values()) / len(skills) <= MAX_DOMAIN_SHARE,
        "licenses": dict(Counter(s["license"] for s in skills).most_common()),
        "sources": len({s["source_repo"] for s in skills}),
        "prefix_clusters": prefix_cluster_density(skills, final_vectors),
        "corpus_version": file_sha256(SKILLS),
    }
    run = start_run(
        "curation",
        config={
            **asdict(params),
            "pair_judge_min_cosine": PAIR_JUDGE_MIN_COSINE,
            "max_domain_share": MAX_DOMAIN_SHARE,
            "embedding_model": DEFAULT_MODEL,
            "llm_model": client.model,
            "snapshot_version": file_sha256(SNAPSHOT),
            "raw_version": file_sha256(RAW),
        },
        corpus_path=SKILLS,
        evalset_path=None,
    )
    run.write_metrics(metrics)
    return run.path
