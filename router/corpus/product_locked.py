"""Product-locked skills: selectable only by naming the product or hosted service that delivers
the capability. They are explicit-bucket only (evals/LABELING.md section 3).

qwen pre-screens every skill into a review sheet; a person confirms each row; the confirmed
list is frozen in corpus/product_locked.json. Kept outside corpus/skills.jsonl so confirming
the list doesn't change the corpus version that earlier runs record.
"""

import csv
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from router.config import REPO_ROOT
from router.corpus.curate_llm import WORKERS, _call, _system_user
from router.llm import LLMClient
from router.runs import file_sha256, start_run

SKILLS = REPO_ROOT / "corpus" / "skills.jsonl"
REVIEW_CSV = REPO_ROOT / "corpus" / "review" / "product_locked.csv"
FROZEN = REPO_ROOT / "corpus" / "product_locked.json"
LLM_CACHE = REPO_ROOT / "corpus" / "raw" / "llm_cache.jsonl"

SYSTEM = """You classify one AI-agent skill.

A skill is PRODUCT-LOCKED if the product or hosted service it uses is the means by which the
capability is delivered, so a user would only pick this skill by naming that product. Examples:
generating video through RunComfy or Kling, images through Higgsfield, web search through
Tavily or Exa, scraping through Firecrawl, or a vendor's own agent tool like Caveman or Agentix.
Other tools could deliver the same result; this skill is tied to one of them.

A skill is NOT product-locked if the product is the user's existing environment that the work
targets: a framework, cloud, database or app the user already has (for example "our Vue app",
"our Azure subscription", "we use Clerk", "our Notion workspace"), or if the skill is generic
and names no product.

Reply with only a JSON object, no other text:
{"product_locked": true | false, "product": "<product name or empty>",
 "reason": "<at most 15 words>"}"""

COLUMNS = [
    "id",
    "repo",
    "description_original",
    "qwen_locked",
    "qwen_product",
    "qwen_reason",
    "decision",
    "note",
]


def request(s: dict[str, Any]) -> tuple[list[dict[str, str]], int]:
    user = f"repo: {s['source_repo']}\nname: {s['name']}\ndescription: {s['description_original']}"
    return _system_user(SYSTEM, user), 60


def prescreen(client: LLMClient, allow_dirty: bool = False) -> Path:
    skills = [json.loads(line) for line in SKILLS.read_text().splitlines() if line.strip()]
    run = start_run(
        "product_locked_prescreen",
        config={"llm_model": client.model, "prompt": SYSTEM},
        corpus_path=SKILLS,
        evalset_path=None,
        allow_dirty=allow_dirty,
    )

    def one(s: dict[str, Any]) -> dict[str, Any]:
        data = _call(client, request(s)) or {}
        locked = data.get("product_locked")
        return {
            "id": s["id"],
            "repo": s["source_repo"],
            "description_original": s["description_original"],
            "qwen_locked": "yes" if locked is True else "no" if locked is False else "parse_failed",
            "qwen_product": str(data.get("product", "")),
            "qwen_reason": str(data.get("reason", "")),
            "decision": "",
            "note": "",
        }

    with ThreadPoolExecutor(WORKERS) as pool:
        rows = list(pool.map(one, skills))
    # Flagged rows first so the reviewer sees them together.
    rows.sort(key=lambda r: (r["qwen_locked"] != "yes", r["repo"], r["id"]))
    REVIEW_CSV.parent.mkdir(parents=True, exist_ok=True)
    with REVIEW_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    run.write_metrics(
        {
            "verdicts": dict(Counter(r["qwen_locked"] for r in rows)),
            "flagged_by_repo": dict(
                Counter(r["repo"] for r in rows if r["qwen_locked"] == "yes").most_common()
            ),
            "review_csv_sha256": file_sha256(REVIEW_CSV),
        }
    )
    return run.path


def flagged_ids() -> set[str]:
    """qwen's provisional flags, used only until the list is confirmed and frozen."""
    with REVIEW_CSV.open(newline="") as f:
        return {r["id"] for r in csv.DictReader(f) if r["qwen_locked"] == "yes"}


def freeze() -> list[str]:
    """Every row needs decision locked|not_locked; write the confirmed list."""
    with REVIEW_CSV.open(newline="") as f:
        rows = list(csv.DictReader(f))
    bad = [r["id"] for r in rows if r["decision"].strip() not in ("locked", "not_locked")]
    if bad:
        raise ValueError(f"{len(bad)} rows without locked/not_locked, e.g. {bad[:3]}")
    locked = sorted(r["id"] for r in rows if r["decision"].strip() == "locked")
    FROZEN.write_text(
        json.dumps(
            {
                "frozen_at": datetime.now(UTC).isoformat(),
                "review_csv_sha256": file_sha256(REVIEW_CSV),
                "corpus_version": file_sha256(SKILLS),
                "product_locked": locked,
            },
            indent=2,
        )
        + "\n"
    )
    return locked


def load_product_locked(path: Path = FROZEN) -> set[str]:
    return set(json.loads(path.read_text())["product_locked"]) if path.exists() else set()
