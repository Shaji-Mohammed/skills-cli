import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from router.llm import LLMClient
from router.spend import SpendLimitExceeded, SpendTracker

MODEL = "openai/gpt-4o-mini"
MESSAGES = [{"role": "user", "content": "hello"}]


def fake_http(response: dict[str, Any], status: int = 200) -> tuple[httpx.Client, list[Any]]:
    seen: list[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(status, json=response)

    return httpx.Client(transport=httpx.MockTransport(handler)), seen


def ok_response(usage: dict[str, Any]) -> dict[str, Any]:
    return {"choices": [{"message": {"content": "hi"}}], "usage": usage}


def test_chat_settles_reported_cost(tmp_path: Path) -> None:
    tracker = SpendTracker(tmp_path / "ledger.jsonl", 5.0)
    http, seen = fake_http(
        ok_response({"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.001})
    )
    result = LLMClient("key", tracker, http).chat(MODEL, MESSAGES, purpose="test")
    assert result.text == "hi"
    assert tracker.total() == pytest.approx(0.001)
    assert seen[0]["usage"] == {"include": True}


def test_chat_falls_back_to_price_table(tmp_path: Path) -> None:
    tracker = SpendTracker(tmp_path / "ledger.jsonl", 5.0)
    http, _ = fake_http(ok_response({"prompt_tokens": 1_000_000, "completion_tokens": 0}))
    LLMClient("key", tracker, http).chat(MODEL, MESSAGES, purpose="test")
    assert tracker.total() == pytest.approx(0.15)


def test_http_error_keeps_reservation(tmp_path: Path) -> None:
    tracker = SpendTracker(tmp_path / "ledger.jsonl", 5.0)
    http, _ = fake_http({"error": "boom"}, status=500)
    with pytest.raises(httpx.HTTPStatusError):
        LLMClient("key", tracker, http).chat(MODEL, MESSAGES, purpose="test")
    assert tracker.total() > 0


def test_refused_call_never_hits_network(tmp_path: Path) -> None:
    tracker = SpendTracker(tmp_path / "ledger.jsonl", 0.0)
    http, seen = fake_http(ok_response({}))
    with pytest.raises(SpendLimitExceeded):
        LLMClient("key", tracker, http).chat(MODEL, MESSAGES, purpose="test")
    assert seen == []


def test_unpriced_model_rejected(tmp_path: Path) -> None:
    tracker = SpendTracker(tmp_path / "ledger.jsonl", 5.0)
    http, seen = fake_http(ok_response({}))
    with pytest.raises(ValueError):
        LLMClient("key", tracker, http).chat("some/other-model", MESSAGES, purpose="test")
    assert seen == []


def test_missing_key_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        LLMClient("", SpendTracker(tmp_path / "ledger.jsonl", 5.0))
