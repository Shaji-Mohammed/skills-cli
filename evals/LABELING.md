# Eval labeling rules

These rules were fixed on 2026-10-21, before any prompt was generated. Changing them after
generation starts requires relabeling every prompt.

## 1. REQUIRED vs HELPFUL

A skill is **REQUIRED** for a prompt if both are true:

1. The prompt explicitly asks for, or clearly implies, the deliverable, file format, platform,
   tool or action that the skill provides.
2. The task can't be done to spec without that capability.

A skill is **HELPFUL** if it would improve the result or is a reasonable next step, but the
task can be done to spec without it. HELPFUL skills are **not** in the gold set; returning them
counts as over-routing.

The gold set is the set of REQUIRED skills. It can be empty: for a no-skill prompt, returning
nothing is an exact match and returning any skill is wrong.

### Tie-breakers, applied in order

1. **Process skills** (brainstorming, debugging, TDD, grilling, code review and similar) are
   REQUIRED only when the prompt asks for that process ("write the tests first", "grill me on
   this", "find the root cause"). A skill whose description says "use before any creative work"
   doesn't make itself required. Without an explicit request, process skills are HELPFUL.
   A prompt that needs a process skill **describes the process instead of naming the skill**
   ("ask me hard questions about this plan until every decision is settled", not "grill me"),
   so it still passes rule D2.
2. **Most specific skill wins.** If an umbrella or router skill (for example `firecrawl` or
   `remotion-best-practices`) and a specific skill (`firecrawl-scrape`) both cover the task, the
   specific skill is gold. The umbrella is accepted only through an equivalence set (section 2),
   if one is approved.
3. **Platform named, platform required.** If the prompt names a platform or vendor ("on
   Cloudflare", "with Prisma"), that vendor's skill is gold over a generic one. If no platform
   is named and several vendors' skills fit equally, the prompt is ambiguous: rewrite it to name
   one, or drop it under rule D5.
4. **Reading is not converting.** A skill for a format is REQUIRED when the task produces or
   edits that format, or must read it and no other gold skill can. A second skill that could
   also read the input (for example `firecrawl-parse` for a PDF) is HELPFUL.
5. **One skill per capability.** If two gold skills would provide the same capability, keep one
   and add the other through an equivalence set, not to the gold set.

## 2. Equivalence sets

`evals/equivalence.json` lists groups of skills that are interchangeable for routing.
Labels name one member (the canonical one). The **equivalence-aware** metric accepts any member
in its place. The **strict** metric accepts only the canonical member. Returning two members of
the same set when gold needs one counts as over-routing in both metrics.

Candidates come from skill text only (`router evals equivalence-candidates`):

- the 28 `overlap` pairs in `corpus/overlaps.jsonl`;
- pairs below the 0.85 pair-judge band, for example `mattpocock/skills/tdd` ~
  `addyosmani/agent-skills/test-driven-development`: each skill's nearest skills from other
  repos, same-repo pairs with related names, alias pairs (a description that says it's
  deprecated or renamed) and umbrella pairs (a router skill and the skills it routes to).

Review rules (applied in the 2026-10-21 review):

- **Accept** a pair only if, for any realistic prompt where one skill is REQUIRED, the other
  serves the user equally well.
- **Umbrella ~ specific: reject** (tie-breaker 2: the specific skill is gold).
- **Different hosted services or platforms: reject** (tie-breaker 3), for example Firecrawl vs
  ScrapeGraph, RunComfy vs Higgsfield, an Azure-specific skill vs a generic one. Local tools
  that produce the same result are accepted (agent-browser, browser-act, browser-use).
- **Design-quality skills stay distinct by policy** (frontend-design, impeccable,
  ui-ux-pro-max and similar). This is a known over-routing risk; see `docs/DECISIONS.md`.
- **Canonical member:** the most-installed member, except `prisma/skills/prisma-postgres-setup`
  for the alias set and `shadcn/ui/shadcn` for the shadcn set.

qwen3.5:9b pre-screens the pool with the curation pair prompt. A person reviews every pair it
calls `same` or `overlap`, every alias and umbrella pair, and a random sample of pairs it called
`distinct` (to check what the pre-screen misses). If 3 or more of the 20 spot-check pairs had
been accepted, the sheet would have been widened before freezing; 0 were. Accepted pairs are
merged into sets, and the sets are frozen in `evals/equivalence.json` before any prompt is
generated.

## 3. Buckets and mix

Every prompt has one primary bucket. A separate `multi_skill` flag marks gold sets with 2+
skills. Per-bucket results report explicit, implied and no-skill, plus multi-skill as its own
slice (which overlaps the first two).

| Bucket | Definition | Target | Source |
| --- | --- | ---: | --- |
| explicit | Names the format, tool or platform | 200 (50%) | model_drafted |
| implied | Describes only the goal; no format, tool or platform named | 120 (30%) | 50 hand_written, 70 model_drafted |
| no-skill | Answerable without any skill, or out of scope for the corpus | 80 (20%) | model_drafted |
| *multi-skill (slice)* | Gold set of 2 or 3 skills | ≥ 100 (≥ 25%) | both |

Every result is reported for `hand_written` and `model_drafted` separately as well as combined,
because qwen3.5:9b both drafts prompts and acts as the fallback.

**Product-locked skills are explicit-bucket only.** A skill is product-locked if a prompt can
only select it by naming its product or hosted service, because the product is the means by
which the capability is delivered (RunComfy, Higgsfield, Tavily/Exa, Stripe Projects, Caveman,
Agentix, shoehorn, Kling, Firecrawl and similar). Product-locked skills never appear in implied
prompts, hand-written or drafted. The confirmed list is `corpus/product_locked.json`: qwen
pre-screens every skill, and a person confirms each flag.

**Environment cues are allowed in implied prompts; capability names are not.** An implied
prompt may name what the user already has ("our Vue app", "our Azure subscription", "we use
Clerk"), but not what the skill provides: no format, tool, product or service that does the
job. This applies to hand-written prompts and qwen's implied drafts alike.

- Multi-skill target: about 70 explicit and 30+ implied. About 3 in 4 are pairs and 1 in 4 are
  triples.
- Coverage: each skill appears in at most 3 gold sets, and at least 250 distinct skills appear
  in some gold set, so a few vendors can't dominate the score.
- Multi-skill combinations must occur together in real workflows. They come from cross-references
  in skill text (for example "for email sequences, see emails") and a hand-curated pair list
  you approve. They're never sampled at random.

## 4. Generation (model-drafted prompts)

- **Input:** the gold skill's SKILL.md body (frontmatter removed). Never the rewritten
  description. I'd also avoid the original description; see open question 1.
- **Model:** qwen3.5:9b, with a fixed seed and temperature 0, and responses cached and committed.
- **Every prompt records** its `source` (`model_drafted` or `hand_written`), bucket, gold set,
  the skill it was drafted from and its `review_status`.

## 5. Review

The split (section 7) happens first, then review:

- **test:** every item is reviewed in full (prompt, gold, HELPFUL exclusions) before the test
  file is locked. `review_status = full`.
- **dev:** every multi-skill item, plus a random 30% of the rest (fixed seed), is reviewed.
  `review_status = sampled` for these, `unreviewed` for the others. The README reports the
  counts of each status.

The review sheet (CSV) has one row per prompt: prompt, proposed gold, HELPFUL exclusions, a
`decision` column (`accept`, `edit`, `drop`), an `edited_gold` column and a **`missing_skill`**
column for any REQUIRED skill not in the proposed gold.

**The review aid never shows only what a router would retrieve.** If it did, gold labels would
drift toward retrievable skills. Each row lists the **union** of the BM25 top 10 and the
embedding top 10 for the prompt (each computed independently, untuned) next to the proposed
gold. The `missing_skill` column takes any skill id from the full corpus, listed or not.

## 6. Degenerate prompts (dropped before any router runs)

A prompt is dropped if any of these hold. Drops are logged with the rule that applied.

- **D1.** Fewer than 6 words or more than 120 words.
- **D2.** Contains a gold skill's multi-token name verbatim (for example "firecrawl-scrape").
  Single-token names that are also formats or products ("pdf", "seo") are allowed in explicit
  prompts.
- **D3.** Shares a 5-word sequence (lowercased, punctuation removed) with the original or
  rewritten description of any gold skill.
- **D4.** Description-embedding cosine ≥ 0.95 with an earlier prompt in the eval set (the later
  one is dropped).
- **D5.** The label reviewer marks it ambiguous under the tie-breakers in section 1.

None of these criteria look at router output.

## 7. Splits

- All prompts, including the hand-written ones, are pooled after the D1–D4 filters, then split
  with a fixed seed into `dev` (250) and `test` (150), stratified by bucket × source ×
  multi-skill.
- Expected test sizes per bucket: about 75 explicit, 45 implied (about 19 hand-written and 26
  model-drafted), 30 no-skill and 38 or more multi-skill. Intervals per bucket will be wide:
  with 45 items, a Wilson 95% interval around 80% runs from about 66% to 89%.
- After the test review, `evals/test.jsonl` is written once. Its SHA-256 goes in the README,
  and it isn't opened again until Phase 7. Scripts that touch it print only counts and the
  hash, never rows.
