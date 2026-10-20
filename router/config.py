"""Paths and settings shared across the project."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
# Committed on purpose: the $5 cap is a lifetime budget, so the ledger must survive clones.
SPEND_LEDGER = RESULTS_DIR / "spend_ledger.jsonl"


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str
    spend_limit_usd: float
    spend_confirm_usd: float


def load_settings() -> Settings:
    load_dotenv(REPO_ROOT / ".env")
    return Settings(
        openrouter_api_key=os.environ.get("OPENROUTER_API_KEY", ""),
        spend_limit_usd=float(os.environ.get("SPEND_LIMIT_USD", "5.00")),
        spend_confirm_usd=float(os.environ.get("SPEND_CONFIRM_USD", "0.50")),
    )
