# Skill Router

Given a natural-language prompt, return every skill it needs from a curated corpus of public
skills. Three-stage cascade: graph-expanded hybrid retrieval, then calibrated scoring, then an LLM
fallback for uncertain candidates only. See [ROADMAP.md](ROADMAP.md) for the plan.

**Status:** Phase 1 done: a curated corpus of 500 skills ([corpus/CURATION.md](corpus/CURATION.md)).
No routing results yet.

## Setup

```sh
uv sync
cp .env.example .env
uv run pre-commit install
uv run router --help
```

The LLM steps run locally on [Ollama](https://ollama.com) with `qwen3.5:9b`
(`ollama pull qwen3.5:9b`). Rebuilding the corpus replays the committed response cache, so
Ollama is only needed when the inputs change:

```sh
uv run router corpus fetch-leaderboard   # snapshot skills.sh (already committed)
uv run router corpus collect             # clone source repos into .cache/
uv run router corpus curate              # apply the curation rules -> corpus/skills.jsonl
uv run router corpus report              # write corpus/CURATION.md from the latest run
```

## Corpus

500 skills chosen from the skills.sh all-time install leaderboard, then curated (see
[corpus/CURATION.md](corpus/CURATION.md)). "Popular" here means most installed, not highest
quality: install counts can be inflated, and nothing in the ranking checks whether a skill
works.

**Licensing.** Skills keep their authors' licenses: 310 are MIT, 118 Apache-2.0, and the rest are
listed in CURATION.md (49 have no license file; 4 Anthropic document skills are proprietary).
This repo stores only each skill's name, description, path, source commit and license, plus
short rewritten descriptions for 55 skills. It doesn't store any SKILL.md bodies; the collector
reads them from local clones in `.cache/` (git-ignored). Follow each source repo's license if
you reuse its skills.

## Results policy

- Every number in this README comes from a saved run in `results/` whose `run.json` records the
  config, corpus version, eval-set version and git commit.
- Results are reported with 95% Wilson confidence intervals, by bucket and separately for
  hand-written and model-drafted prompts (qwen3.5:9b drafts prompts and is also the fallback).
- The test set is locked until the final evaluation. Its SHA-256 will be recorded here when it's
  created in Phase 2.
- Planned test split: 150 prompts, about 75 explicit, 45 implied (about 19 hand-written), 30
  no-skill and at least 38 multi-skill. Per-bucket intervals will be wide: with 45 items, an
  80% score has a 95% interval of about 66–89%. Labeling rules: [evals/LABELING.md](evals/LABELING.md).
- All paid API calls go through a spend tracker with a hard $5 cap: `uv run router spend`.
