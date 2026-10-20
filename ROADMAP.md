# Skill Router — From-Scratch Build Roadmap

Oct 20, 2026

## Overview

Build a new skill router from an empty repo: given a prompt, return every skill it needs from a corpus of 400–600 public skills, at low latency, low cost and as close to 95% exact-set match as the data allows.

**Architecture (three-stage cascade):**

1. **Graph shortlist:** hybrid retrieval (embeddings + BM25) seeds candidates, and a skill graph adds what text matching misses (implied needs, companion skills).
2. **Jev scoring:** TypeSafe's Jev scores each shortlisted skill as required or not, with a calibrated probability.
3. **LLM fallback:** only candidates Jev is unsure about go to a cheap LLM.

**Lessons carried over from v1 (the code is new, the lessons aren't):**

- Corpus quality caps accuracy: near-duplicate skills and jargon descriptions can't be routed by any method. Curate the corpus *before* building the router.
- Over-routing (correct skills plus extras) was the main failure, so the final stage must be good at saying no.
- Implied-need prompts ("something for the board meeting") were the weakest bucket at 50%.
- Tuning on a test set's failures inflates the score. Keep one test set untouched until the end.

**Cost:** under $5, enforced by a spend tracker with a hard halt.

## Phase 0: Setup

- [ ] **Apply for Jev early access today.** Until a key arrives, Phase 5 runs with a local cross-encoder (`bge-reranker-base`) behind the same interface, so nothing blocks.
- [ ] New repo layout: `router/` (stages), `corpus/`, `evals/`, `graph/`, `scripts/`, `results/`
- [ ] Python 3.12, `uv`, ruff, mypy, pytest, pre-commit; Typer for the CLI
- [ ] Spend tracker: every paid API call logs its cost to a file and halts the program at $5
- [ ] Results convention: every eval run saves its config, corpus version, eval-set version and git commit to `results/`
- [ ] GitHub Actions: lint, unit tests, and a fast deterministic eval (Stage 1 recall) on every PR

## Phase 1: Corpus

Done means: 400–600 distinct, routable skills in one schema, with documented curation decisions.

**Collect**

- [ ] Pull skills from public repos (for example `anthropics/skills`, community skill collections); record source and license per skill
- [ ] Normalize to `{id, name, description, source_repo, source_path, license}`

**Curate (the step that sets your ceiling)**

- [ ] Deduplicate by name and by near-identical description (cosine above about 0.92), keeping the best-described copy
- [ ] **Cluster density check:** for each name-prefix or topic cluster, compute the mean pairwise similarity. Collapse or drop clusters where skills are indistinguishable.
- [ ] Flag descriptions that say *when* a skill triggers rather than *what it does*; rewrite only those, keeping the original in `description_original`
- [ ] Check domain spread: no single domain above about 35% of the corpus
- [ ] Write `corpus/CURATION.md`: every rule applied, and how many skills each rule removed

## Phase 2: Eval set

Done means: dev and test sets with multi-skill labels, and the test set locked.

**Labeling rule (decide it before writing any prompt)**

- [ ] A skill is REQUIRED if the task explicitly asks for or clearly implies the deliverable, format, platform or action it provides, and the task can't be done to spec without it. Adjacent-but-optional skills are HELPFUL and excluded.
- [ ] **Equivalence sets** for remaining near-duplicates: picking any member counts as correct in the equivalence-aware metric

**Prompts (about 400 total)**

- [ ] About 50% explicit (names the format or tool), 30% implied (describes the goal only), 20% no-skill-needed or out of scope
- [ ] At least 25% multi-skill, built from skill pairs and triples that genuinely occur together in workflows
- [ ] LLM-drafted, but you review every label. Hand-write at least 40 of the implied prompts yourself.
- [ ] Drop degenerate prompts by a criterion fixed in advance, not by whether the router gets them right

**Splits**

- [ ] `dev` (about 250): tune on this freely
- [ ] `test` (about 150): **locked**. Run it once at the end. Store a hash of the file in the README so readers can see it didn't change.

**Metrics (write `evals/score.py` now, before any router exists)**

| Metric | What it measures |
| --- | --- |
| Exact-set match (strict) | Headline: the returned set equals the gold set |
| Exact-set match (equivalence-aware) | Same, counting near-duplicate swaps as correct |
| Stage 1 recall at cap | Were all gold skills in the shortlist? |
| Precision / recall per prompt | Separates over-routing from misses |
| Failure taxonomy | MISS_STAGE1, OVERROUTING, WRONG_SCORER, ABSTAIN |
| Per-bucket accuracy | Explicit, implied, multi-skill, no-skill |
| Latency p50/p95 per path, fallback rate, cost per 1K | The constraints |

Report everything with 95% confidence intervals (Wilson intervals are fine).

## Phase 3: Baseline retrieval

Done means: a measured baseline on dev that every later stage must beat.

- [ ] Embeddings: compare `all-MiniLM-L6-v2` and `bge-small-en-v1.5` (both local and free)
- [ ] BM25 over name + description (`rank_bm25`)
- [ ] Hybrid score with a weight sweep (for example 0.5–0.9 on embeddings)
- [ ] Relative gate (keep candidates above a fraction of the top score) plus a hard cap
- [ ] Record on dev: Stage 1 recall at caps of 10/20/40, median candidate count, p50/p95 latency
- [ ] Simple end-to-end baseline: take the top-k above the gate as the answer, so later stages have a number to beat

## Phase 4: Skill graph and shortlist

Done means: Stage 1 recall on dev rises clearly above the Phase 3 baseline without the median candidate count blowing up. Aim for 97–98% recall at the cap, since you can't reach 95% end to end otherwise.

**Build the graph (JSON + `networkx`)**

- [ ] **Intent → skill edges:** for each skill, generate 3–5 "a user would say…" sentences **from its description only**, never from eval prompts. Spot-check 50; drop vague ones.
- [ ] **Co-occurrence edges:** skill pairs that appear together in **dev** labels, plus hand-curated workflow pairs (pdf ↔ xlsx), marked as curated
- [ ] **Taxonomy edges:** cluster skills into 20–40 domains, name the clusters with an LLM, review the names by hand

**Shortlist algorithm**

1. Seeds: hybrid retrieval top-k, plus skills reached through the best-matching intent sentences.
2. Expand: personalized PageRank from the seeds (compare against a simple 1-hop expansion with decayed weights).
3. Rank by combined score; cap at 20–40.

**Ablations on dev (one change at a time)**

- [ ] Seeds only → + intent edges → + co-occurrence → + taxonomy
- [ ] For each: Stage 1 recall, implied-bucket recall, median candidates, latency

## Phase 5: Jev scorer and calibration

Done means: every shortlisted skill gets a calibrated "required" probability, and you've measured how good the calibration really is.

**Scorer interface**

- [ ] A `Scorer` protocol: (prompt, candidates) → probability per candidate
- [ ] Three implementations: **Jev**, **`bge-reranker-base`** (local, free) and **gpt-4o-mini** (ask for a JSON list of required skills)

**Jev query (check the exact schema in TypeSafe's docs)**

- [ ] State: the prompt plus each candidate's name and description
- [ ] Output: one boolean "required" decision per candidate with its probability, scored independently, so multi-skill prompts work naturally
- [ ] Put your REQUIRED vs HELPFUL rule in the question text, so Jev scores the same definition your labels use

**Measure on dev for each scorer**

- [ ] Exact-set match at the best single threshold
- [ ] **Calibration:** a reliability diagram and ECE. Does 0.9 confidence mean about 90% correct? This tests TypeSafe's main claim.
- [ ] Latency p50/p95 measured from Toronto (their published numbers come from the US West Coast)
- [ ] Cost per 1K queries

## Phase 6: Cascade and fallback

Done means: one configuration, tuned on dev, with the best trade-off between accuracy, fallback rate and cost.

**Decision rule per candidate**

- p ≥ `hi` → include
- p ≤ `lo` → exclude
- in between → send **only those uncertain candidates** to gpt-4o-mini, with the prompt and the already-accepted skills as context

**Also**

- [ ] **Fast path:** if one candidate dominates the shortlist and Jev is confident, skip everything else
- [ ] **Abstain:** if nothing clears `lo` after fallback, return "no confident match" rather than guessing
- [ ] Grid over `hi` and `lo` on dev; plot exact-set match against fallback rate. This curve is a strong README figure.
- [ ] Pick the point that meets your accuracy target with the lowest fallback rate, then freeze the config

## Phase 7: Final test, README and resume

- [ ] Run the frozen config **once** on the locked test set
- [ ] Report, with confidence intervals:
    - strict and equivalence-aware exact-set match
    - per-bucket accuracy (explicit, implied, multi-skill, no-skill)
    - the failure taxonomy
    - latency p50/p95 per path, fallback rate, cost per 1K queries
- [ ] **Ablation table:** baseline → + graph → + Jev → + fallback (measured on dev)
- [ ] **Scorer comparison:** Jev vs cross-encoder vs gpt-4o-mini, with calibration plots
- [ ] An honest "why not 95%" section if you fall short. Your v1 README showed this kind of honesty reads well.
- [ ] Update the resume bullet with the real numbers

**Resume bullet template (real numbers only):**

> Built a 3-stage skill router over N skills (graph-expanded hybrid retrieval → calibrated Jev scoring → LLM fallback on uncertain cases), reaching X% exact-set match on a locked test set with LLM calls on Y% of queries, p95 latency of Z ms, and $W per 1K queries.

## Risks and scope-cut rules

| Risk | Early sign | What to do |
| --- | --- | --- |
| Jev access delayed | No key by the time Phase 5 starts | Run Phases 5–6 with the cross-encoder; add Jev later as a scorer comparison |
| Corpus too dense | Many clusters above the similarity threshold | Collapse more aggressively; a smaller routable corpus beats a big ambiguous one |
| Graph adds noise | Over-routing rises on dev | Keep intent edges only; lower the cap |
| Contamination | Dev and test diverge, or you're tempted to peek at test | Test stays locked; all edges come from descriptions and dev only |
| Poor Jev calibration | Reliability curve far off the diagonal | Recalibrate on dev (isotonic regression) or widen the fallback band |
| Scope creep | Phase 3 drags on without a scored baseline | Ship baseline + Jev without the graph; the graph becomes "next step" |

**If you need a smaller version:** Phases 0–3, then Jev scoring directly on the hybrid shortlist (skip the graph). Calibrated scoring alone targets over-routing, the biggest failure from v1.
