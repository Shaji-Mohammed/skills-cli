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
The real cost comes from OpenRouter's `usage.cost` field; the price table is used only if that
field is missing.

**Run manifests record `git_dirty`.** A run from uncommitted code can't be reproduced, so only
runs with `git_dirty: false` are quoted in the README or in phase summaries.

**The test split is locked in code** (`router/splits.py`). Loading it requires
`allow_test=True`, so any use shows up in a diff.

**The Stage 1 recall job is added to CI in Phase 3.** There's no retriever to evaluate until then.
