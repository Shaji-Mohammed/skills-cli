"""Eval split loading, with the test split locked behind an explicit flag.

The test set is run exactly once, in Phase 7. Any code path that loads it must pass
allow_test=True, which makes accidental use visible in a diff.
"""

import json
from pathlib import Path
from typing import Any

from router.config import REPO_ROOT

EVALS_DIR = REPO_ROOT / "evals"


class TestSetLocked(RuntimeError):
    """Raised when the locked test split is requested without allow_test=True."""

    __test__ = False  # stop pytest from trying to collect this as a test class


def split_path(name: str) -> Path:
    return EVALS_DIR / f"{name}.jsonl"


def load_split(name: str, allow_test: bool = False) -> list[dict[str, Any]]:
    if name == "test" and not allow_test:
        raise TestSetLocked("The test split is locked until Phase 7")
    with split_path(name).open() as f:
        return [json.loads(line) for line in f if line.strip()]
