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

## Results policy

- Every number in this README comes from a saved run in `results/` whose `run.json` records the
  config, corpus version, eval-set version and git commit.
- Results are reported with 95% Wilson confidence intervals, by bucket.
- The test set is locked until the final evaluation. Its SHA-256 will be recorded here when it's
  created in Phase 2.
- All paid API calls go through a spend tracker with a hard $5 cap: `uv run router spend`.
