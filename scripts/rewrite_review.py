"""Measure the rewrite flag on a stratified sample: 10 kept + 10 reverted rewrites.

    uv run python scripts/rewrite_review.py sample   # writes the blind labeling sheet
    uv run python scripts/rewrite_review.py score    # scores hand labels, saves a results run

The sheet hides which group each item came from and shuffles them, so the hand label
("is the original trigger-only?") is made without knowing the model's decision.
"""

import json
import random
import sys
from pathlib import Path

from router.config import REPO_ROOT
from router.runs import start_run
from router.stats import wilson

DECISIONS = REPO_ROOT / "corpus" / "raw" / "rewrite_decisions.jsonl"
REVIEW_DIR = REPO_ROOT / "corpus" / "review"
SHEET = REVIEW_DIR / "rewrite_sample.jsonl"  # blind: id, original, label (filled by hand)
KEY = REVIEW_DIR / "rewrite_sample_key.jsonl"  # which group each sampled id came from
SEED = 20261021
PER_GROUP = 10


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def sample() -> None:
    decisions = read_jsonl(DECISIONS)
    rng = random.Random(SEED)
    kept = rng.sample([d for d in decisions if d["kept"]], PER_GROUP)
    reverted = rng.sample([d for d in decisions if not d["kept"]], PER_GROUP)
    items = kept + reverted
    rng.shuffle(items)
    write_jsonl(
        SHEET,
        [
            {"id": d["id"], "description_original": d["description_original"], "label": None}
            for d in items
        ],
    )
    write_jsonl(KEY, [{"id": d["id"], "kept": d["kept"], "rewrite": d["rewrite"]} for d in items])
    print(f"Wrote {len(items)} items to {SHEET.relative_to(REPO_ROOT)}")


def score() -> None:
    labels = {r["id"]: r["label"] for r in read_jsonl(SHEET)}
    if any(v not in ("trigger_only", "states_what") for v in labels.values()):
        sys.exit("Every item needs label 'trigger_only' or 'states_what'")
    key = {r["id"]: r["kept"] for r in read_jsonl(KEY)}

    kept = [i for i in key if key[i]]
    reverted = [i for i in key if not key[i]]
    kept_correct = sum(labels[i] == "trigger_only" for i in kept)
    reverted_correct = sum(labels[i] == "states_what" for i in reverted)
    # The first-pass flag marked all 20 as trigger-only, so its precision on the sample is the
    # share labeled trigger_only. The sample is 10/10 by design, not proportional to 55/40.
    first_pass_correct = sum(v == "trigger_only" for v in labels.values())

    def ci(k: int, n: int) -> dict[str, object]:
        lo, hi = wilson(k, n)
        return {"k": k, "n": n, "rate": round(k / n, 4), "wilson95": [round(lo, 4), round(hi, 4)]}

    run = start_run(
        "rewrite_review",
        config={"seed": SEED, "per_group": PER_GROUP, "labeler": "Claude, blind to group"},
        corpus_path=REPO_ROOT / "corpus" / "skills.jsonl",
        evalset_path=None,
    )
    metrics = {
        "strict_check_precision_on_kept": ci(kept_correct, len(kept)),
        "strict_check_correct_reverts": ci(reverted_correct, len(reverted)),
        "first_pass_flag_precision_on_sample": ci(first_pass_correct, len(labels)),
    }
    run.write_metrics(metrics)
    print(json.dumps(metrics, indent=2))
    print(f"Saved {run.path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    {"sample": sample, "score": score}[sys.argv[1]]()
