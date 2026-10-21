import json
from pathlib import Path
from typing import Any

import httpx

from router.llm import LLMClient, ResponseCache, extract_json

MESSAGES = [{"role": "user", "content": "hello"}]


def fake_http(reply: str) -> tuple[httpx.Client, list[Any]]:
    seen: list[Any] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        body = {"message": {"content": reply}, "prompt_eval_count": 7, "eval_count": 3}
        return httpx.Response(200, json=body)

    return httpx.Client(transport=httpx.MockTransport(handler)), seen


def test_chat_sends_deterministic_non_thinking_request() -> None:
    http, seen = fake_http("hi")
    result = LLMClient("qwen3.5:9b", http=http).chat(MESSAGES, max_tokens=50)
    assert (result.text, result.input_tokens, result.output_tokens) == ("hi", 7, 3)
    assert seen[0]["think"] is False
    assert seen[0]["options"] == {"temperature": 0, "num_predict": 50}
    assert seen[0]["stream"] is False


def test_cache_hit_skips_the_model(tmp_path: Path) -> None:
    http, seen = fake_http("hi")
    LLMClient("m", http=http, cache=ResponseCache(tmp_path / "c.jsonl")).chat(MESSAGES)
    # A fresh cache instance reads the same file, as a later run would.
    second = LLMClient("m", http=http, cache=ResponseCache(tmp_path / "c.jsonl")).chat(MESSAGES)
    assert second.cached
    assert second.text == "hi"
    assert len(seen) == 1


def test_cache_key_depends_on_model(tmp_path: Path) -> None:
    http, seen = fake_http("hi")
    cache = ResponseCache(tmp_path / "c.jsonl")
    LLMClient("m1", http=http, cache=cache).chat(MESSAGES)
    LLMClient("m2", http=http, cache=cache).chat(MESSAGES)
    assert len(seen) == 2


def test_extract_json() -> None:
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! {"a": {"b": 2}} done') == {"a": {"b": 2}}
    assert extract_json("no json here") is None
    assert extract_json("{broken") is None
    assert extract_json("[1, 2]") is None
