# Design decisions

Non-obvious choices, with the reason for each. Newest at the bottom.

## Phase 0

**Spend tracking reserves before the call and settles after.** A pre-call reservation uses a
pessimistic estimate (about 3 characters per token, plus the full `max_tokens` of output at list
price). The call is refused if the reservation would pass the limit, so the $5 cap can't be
overshot. If a call fails or the process crashes before settling, the reservation keeps counting
at its estimate. The ledger can overstate spend but never understate it.

**The spend ledger is committed** (`results/spend_ledger.jsonl`). The $5 cap covers the whole
project, not one machine or one clone, so the ledger has to travel with the repo.

**Only models in `spend.PRICES` can be called.** That guarantees every call has a cost estimate.

**Run manifests record `git_dirty`.** A run from uncommitted code can't be reproduced, so only
runs with `git_dirty: false` are quoted in the README or in phase summaries.

**The test split is locked in code** (`router/splits.py`). Loading it requires
`allow_test=True`, so any use shows up in a diff.

**The Stage 1 recall job is added to CI in Phase 3.** There's no retriever to evaluate until then.

## Phase 1

**The LLM is local: qwen3.5:9b on Ollama, not gpt-4o-mini on OpenRouter.** There's no
OpenRouter account, and a local model costs nothing. The spend tracker stays for paid APIs
(Jev). The 12 reservations in the ledger from 2026-10-20 are requests OpenRouter rejected with
401 (no account), so nothing was billed. They stay in the ledger because it's append-only.

**Thinking is off and JSON is parsed from plain text.** qwen3.5 thinks by default, which took
about 25 seconds for a one-line reply. On this Ollama version the model has no structured-output
mode, so prompts ask for a bare JSON object and `extract_json` takes the first one in the
reply. An unparseable reply always falls back to the conservative choice (keep the skill, keep
its description) and is counted under `parse_failures` in the run.

**Popularity comes from skills.sh all-time installs.** The goal is a corpus of skills people
actually use, not whatever a repo search returns. The leaderboard endpoint is undocumented, so
the snapshot is committed and every later step reads the snapshot, not the live site.

**We collect the top 1,000 to end with 500.** The 500 cap is applied last, by install count, so
slots freed by removed copies and dead entries go to the next most-installed skills.

**Skills missing from the repo's current HEAD are dropped.** Install counts are cumulative, so
renamed or deleted skills stay on the leaderboard. We don't revive them from git history,
because nobody installing today would get them.

**Which copy to keep:** official source first, then the longer description, then installs.
Description length stands in for the roadmap's "best-described copy". Copies usually have
identical text, so in practice the official flag or install count decides.

**Near-duplicates: the 0.92 cutoff stays, and an LLM judges the 0.85–0.92 band.** With
bge-small, descriptions from the same vendor share so much wording that clearly different
skills score 0.85–0.91 (for example deploy vs observability for the same CLI). Lowering the
cutoff would delete routable skills. The LLM judges each pair in the band as `same`,
`overlap` or `distinct`. Only `same` removes a skill; `overlap` pairs are saved to
`corpus/overlaps.jsonl` as candidate equivalence sets for Phase 2. Per-prefix cluster density is
reported as a diagnostic because, on this corpus, it doesn't indicate whether skills can be
told apart.

**Descriptions are rewritten only when they never say what the skill does.** Many descriptions
are trigger-phrase lists ("use when the user says..."). Those that also state a capability are
left alone. The rest are rewritten from the SKILL.md body, and the original is kept in
`description_original`.

**LLM outputs are cached in the repo** (`corpus/raw/llm_cache.jsonl`), so rebuilding the
corpus gives the same result without re-running the model.
