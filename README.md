# Skill Router

Given a natural-language prompt, return every skill it needs from a curated corpus of public
skills. Three-stage cascade: graph-expanded hybrid retrieval, then calibrated scoring, then an LLM
fallback for uncertain candidates only. See [ROADMAP.md](ROADMAP.md) for the plan.

**Status:** Phase 2 done: a curated corpus of 500 skills ([corpus/CURATION.md](corpus/CURATION.md))
and a reviewed eval set with a locked test split. No routing results yet.

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
- Results are reported with 95% Wilson confidence intervals, by bucket and separately by prompt
  source: `model_drafted` (qwen3.5:9b, which is also the fallback) and `claude_drafted`.
- All paid API calls go through a spend tracker with a hard $5 cap: `uv run router spend`.

## Eval set

Labeling rules: [evals/LABELING.md](evals/LABELING.md). Built by
`results/20261022T000223Z_apply_review`.

**The test set is locked** and isn't run until the final evaluation (Phase 7):

```
evals/test.jsonl  sha256 5e49c9221f01fe959662f4e99284cc3ea253fededf9f0675e783569654d4bc46
```

| Split | Items | Explicit | Implied | No-skill | Multi-skill | claude_drafted | Review status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| test | 123 | 67 | 35 | 21 | 24 | 10 | 123 full (113 accepted, 10 edited; 27 dropped) |
| dev | 217 | 111 | 66 | 40 | 44 | 15 | 67 sampled (54 accepted, 13 edited; 17 dropped), 150 unreviewed |

### Known limitations

- **Test is small.** 123 items after review, so per-bucket intervals are wide: with 35 implied
  items, an 80% score has a 95% interval of about 64–90%.
- **No-skill accuracy is optimistic.** Every no-skill prompt is general knowledge or life advice.
  There are no hard negatives (for example, coding questions that need no skill).
- **Near-duplicate filtering misses paraphrases.** Three no-skill dev prompts turned out to be
  near-duplicates of test prompts and were dropped in review; the D4 filter (cosine ≥ 0.95)
  didn't catch them.
- **No prompt was written by a person.** The 25 prompts planned as hand-written were drafted by
  Claude at the reviewer's request (`claude_drafted`); the rest were drafted by qwen3.5:9b. All
  test labels were reviewed by a person, but prompt phrasing reflects two LLMs, so results may
  overstate accuracy on real user phrasing. The roadmap's goal of 40+ hand-written implied
  prompts was not met.
- **Dev is mostly unreviewed.** 150 of 217 dev items kept their drafted labels unchecked.
