# Skill Router

Given a natural-language prompt, return every skill it needs from a curated corpus of public
skills. Three-stage cascade: graph-expanded hybrid retrieval, then calibrated scoring, then an LLM
fallback for uncertain candidates only. See [ROADMAP.md](ROADMAP.md) for the plan.

**Status:** Phase 0 (setup). No results yet.

## Setup

```sh
uv sync
cp .env.example .env   # add OPENROUTER_API_KEY
uv run pre-commit install
uv run router --help
```

## Results policy

- Every number in this README comes from a saved run in `results/` whose `run.json` records the
  config, corpus version, eval-set version and git commit.
- Results are reported with 95% Wilson confidence intervals, by bucket.
- The test set is locked until the final evaluation. Its SHA-256 will be recorded here when it's
  created in Phase 2.
- All paid API calls go through a spend tracker with a hard $5 cap: `uv run router spend`.
