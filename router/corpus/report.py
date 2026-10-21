"""Render corpus/CURATION.md from a saved curation run. Every number comes from the run files."""

import json
from pathlib import Path
from typing import Any

from router.config import REPO_ROOT, RESULTS_DIR

RULES = {
    "not_github_source": "Source isn't a GitHub `owner/repo` (for example a vendor domain), so "
    "there's no public SKILL.md to collect or license to check.",
    "clone_failed": "The GitHub repo couldn't be cloned (private, deleted or renamed).",
    "not_in_current_repo": "No SKILL.md in the repo's current HEAD matches the leaderboard "
    "entry. Install counts are cumulative, so renamed or deleted skills stay listed.",
    "short_description": "Description shorter than {min_description_words} words: too little "
    "text to route on.",
    "non_english": "More than {max_non_latin_ratio:.0%} of the description's characters are "
    "outside Latin scripts. Eval prompts are English.",
    "same_name": "Same skill name as another source's skill. Keep one copy: the official source, "
    "then the longer description, then more installs.",
    "near_duplicate": "Description cosine ≥ {near_duplicate_cosine} ({embedding_model}) with a "
    "skill kept earlier. Same preference order as `same_name`.",
    "indistinguishable": "Description cosine in [{pair_judge_min_cosine}, "
    "{near_duplicate_cosine}) and judged `same` by {llm_model}: no basis to tell them apart.",
    "below_cap": "Outside the {cap} most-installed survivors.",
}


def latest_run(name: str = "curation", results_dir: Path = RESULTS_DIR) -> Path:
    runs = sorted(p for p in results_dir.glob(f"*_{name}") if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"No {name} run in {results_dir}")
    return runs[-1]


def render(run_dir: Path) -> str:
    manifest: dict[str, Any] = json.loads((run_dir / "run.json").read_text())
    metrics: dict[str, Any] = json.loads((run_dir / "metrics.json").read_text())
    cfg = manifest["config"]
    rel = run_dir.relative_to(REPO_ROOT)

    lines = [
        "# Corpus curation",
        "",
        f"Generated from run `{rel}` (git `{manifest['git_commit'][:10]}`"
        f"{', uncommitted changes' if manifest['git_dirty'] else ''}).",
        f"Corpus version (SHA-256 of `corpus/skills.jsonl`): `{metrics['corpus_version']}`.",
        "",
        f"**{metrics['leaderboard_entries']} leaderboard entries → {metrics['final_skills']} "
        f"skills from {metrics['sources']} source repos.**",
        "",
        "Source: the skills.sh all-time install leaderboard, snapshotted in "
        "`corpus/sources/skills_sh_alltime.json`. Rules run in this order; each row counts only "
        "what that rule removed from the skills still remaining. Every removal is listed with its "
        "rule in `corpus/removed.jsonl`.",
        "",
        "| # | Rule | What it removes | Removed | Remaining |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for i, s in enumerate(metrics["steps"], start=1):
        why = RULES[s["rule"]].format(**cfg)
        lines.append(f"| {i} | `{s['rule']}` | {why} | {s['removed']} | {s['remaining']} |")

    nn = metrics["nearest_neighbour_cosine_after_near_duplicate"]
    verdicts = metrics["pair_verdicts"]
    pf = metrics["parse_failures"]
    lines += [
        "",
        "## Similar pairs",
        "",
        "After step 7 (`near_duplicate`), each skill's nearest-neighbour description cosine "
        "had percentiles "
        + ", ".join(f"{k} = {v}" for k, v in nn.items())
        + f". {metrics['pairs_judged']} pairs fell in "
        f"[{cfg['pair_judge_min_cosine']}, {cfg['near_duplicate_cosine']}) and were judged by "
        f"{cfg['llm_model']}: "
        + ", ".join(f"{v} `{k}`" for k, v in sorted(verdicts.items()))
        + ".",
        "",
        "`same` pairs drive step 8 (`indistinguishable`). "
        f"{metrics['overlap_pairs_kept']} `overlap` pairs where both skills survived are saved "
        "in `corpus/overlaps.jsonl` as candidate equivalence sets for the Phase 2 eval. "
        "A reply that couldn't be parsed counts as `parse_failed` and never removes a skill. "
        "All verdicts are in `corpus/raw/pair_judgments.jsonl`.",
        "",
        "## Description rewrites",
        "",
        f"{cfg['llm_model']} flagged {metrics['flagged_trigger_only']} descriptions that say "
        "when to trigger the skill but not what it does. "
        f"{metrics['rewritten_descriptions']} were rewritten from the SKILL.md body; the original "
        "is kept in `description_original`. Nothing else was rewritten.",
        "",
        "## Domain spread",
        "",
        f"Largest domain share: {metrics['max_domain_share']:.1%} (limit "
        f"{cfg['max_domain_share']:.0%}: "
        f"{'OK' if metrics['max_domain_share_ok'] else 'EXCEEDED'}).",
        "",
        "| Domain | Skills |",
        "| --- | ---: |",
        *[f"| {d} | {n} |" for d, n in metrics["domains"].items()],
        "",
        "## Name-prefix cluster density (diagnostic)",
        "",
        "Mean and max pairwise description cosine within each name prefix that has 3 or more "
        "skills. On this corpus a high mean mostly reflects shared vendor wording, so it doesn't "
        "remove anything by itself; the pair judge above decides. See `docs/DECISIONS.md`.",
        "",
        "| Prefix | Skills | Mean cosine | Max cosine |",
        "| --- | ---: | ---: | ---: |",
        *[
            f"| {c['prefix']} | {c['size']} | {c['mean_cosine']} | {c['max_cosine']} |"
            for c in metrics["prefix_clusters"][:15]
        ],
        "",
        "## Licenses",
        "",
        "Licenses come from SKILL.md frontmatter, else the repo's LICENSE file. `unknown` means "
        "neither exists. Only names, descriptions and paths are stored here, not SKILL.md bodies.",
        "",
        "| License | Skills |",
        "| --- | ---: |",
        *[f"| {k} | {v} |" for k, v in metrics["licenses"].items()],
        "",
        f"LLM parse failures: pairs {pf['pairs']}, classify {pf['classify']}, rewrite "
        f"{pf['rewrite']}.",
        "",
    ]
    return "\n".join(lines)
