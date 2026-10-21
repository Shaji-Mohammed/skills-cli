"""Which description text a stage reads: the author's original or the curated rewrite.

Retrieval configs carry `description_source` so Phase 3 can compare both on dev. Eval prompts
are generated from original text only, never from rewrites (see evals/LABELING.md).
"""

from typing import Any, Literal

DescriptionSource = Literal["original", "rewritten"]


def description_for(skill: dict[str, Any], source: DescriptionSource) -> str:
    if source == "original":
        return str(skill["description_original"])
    # "rewritten" means the rewrite where one was kept, else the original.
    return str(skill["description"])
