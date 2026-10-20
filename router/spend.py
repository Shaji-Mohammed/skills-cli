"""Spend tracker: every paid API call is reserved, then settled, in an append-only ledger.

Before a call we append a `reserve` entry with a worst-case estimate; if that would push the
total past the limit, the call is refused. After the call we append a `settle` entry with the
actual cost. An unsettled reservation (crash, network error) keeps counting at its estimate, so
the ledger can only overstate spend, never understate it.
"""

import json
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class SpendLimitExceeded(RuntimeError):
    """Raised when a call would take total spend past the hard limit."""


class ConfirmationRequired(RuntimeError):
    """Raised when a run's estimated cost needs explicit user approval."""


@dataclass(frozen=True)
class Price:
    """USD per million tokens."""

    input_per_mtok: float
    output_per_mtok: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok
        ) / 1_000_000


# OpenRouter list prices. Only models listed here may be called, so every call has an estimate.
PRICES: dict[str, Price] = {
    "openai/gpt-4o-mini": Price(input_per_mtok=0.15, output_per_mtok=0.60),
}


def estimate_tokens(text: str) -> int:
    # Deliberately pessimistic (~3 chars/token vs the usual ~4) so reservations overestimate.
    return len(text) // 3 + 1


class SpendTracker:
    def __init__(self, ledger: Path, limit_usd: float) -> None:
        self.ledger = ledger
        self.limit_usd = limit_usd
        self._lock = threading.Lock()

    def _entries(self) -> list[dict[str, Any]]:
        if not self.ledger.exists():
            return []
        with self.ledger.open() as f:
            return [json.loads(line) for line in f if line.strip()]

    def total(self) -> float:
        """Settled costs plus the estimate of every reservation that was never settled."""
        reserved: dict[str, float] = {}
        settled: dict[str, float] = {}
        for e in self._entries():
            if e["kind"] == "reserve":
                reserved[e["call_id"]] = e["estimate_usd"]
            elif e["kind"] == "settle":
                settled[e["call_id"]] = e["cost_usd"]
        return sum(settled.get(call_id, est) for call_id, est in reserved.items())

    def _append(self, entry: dict[str, Any]) -> None:
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        entry["ts"] = datetime.now(UTC).isoformat()
        with self.ledger.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def reserve(self, model: str, estimate_usd: float, purpose: str) -> str:
        with self._lock:
            current = self.total()
            if current + estimate_usd > self.limit_usd:
                raise SpendLimitExceeded(
                    f"Refusing call: spent ${current:.4f} + estimate ${estimate_usd:.4f} "
                    f"would exceed limit ${self.limit_usd:.2f}"
                )
            call_id = uuid.uuid4().hex
            self._append(
                {
                    "kind": "reserve",
                    "call_id": call_id,
                    "model": model,
                    "estimate_usd": estimate_usd,
                    "purpose": purpose,
                }
            )
            return call_id

    def settle(self, call_id: str, cost_usd: float, usage: dict[str, Any]) -> None:
        with self._lock:
            self._append({"kind": "settle", "call_id": call_id, "cost_usd": cost_usd, **usage})

    def check_run_budget(self, estimate_usd: float, confirm_usd: float, confirmed: bool) -> None:
        """Gate a whole run before it starts: it must fit the limit, and big runs need approval."""
        current = self.total()
        if current + estimate_usd > self.limit_usd:
            raise SpendLimitExceeded(
                f"Run estimate ${estimate_usd:.4f} + spent ${current:.4f} "
                f"exceeds limit ${self.limit_usd:.2f}"
            )
        if estimate_usd > confirm_usd and not confirmed:
            raise ConfirmationRequired(
                f"Run estimate ${estimate_usd:.4f} exceeds ${confirm_usd:.2f}; "
                "re-run with --yes after approval"
            )
