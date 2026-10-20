"""OpenRouter chat client: the only module that makes paid calls, all through SpendTracker."""

import hashlib
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from router.spend import PRICES, SpendTracker, estimate_tokens

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass(frozen=True)
class ChatResult:
    text: str
    cost_usd: float
    input_tokens: int
    output_tokens: int
    cached: bool = False


class ResponseCache:
    """Append-only JSONL of request-hash -> response text.

    Committed alongside the data it produced, so rebuilding the corpus replays the same LLM
    outputs for free instead of re-paying (and getting slightly different answers).
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._entries: dict[str, str] = {}
        if path.exists():
            with path.open() as f:
                for line in f:
                    if line.strip():
                        e = json.loads(line)
                        self._entries[e["key"]] = e["text"]

    @staticmethod
    def key(body: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()

    def get(self, key: str) -> str | None:
        return self._entries.get(key)

    def put(self, key: str, text: str) -> None:
        with self._lock:
            self._entries[key] = text
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps({"key": key, "text": text}, ensure_ascii=False) + "\n")


def request_body(
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    response_format: dict[str, Any] | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if response_format is not None:
        body["response_format"] = response_format
    return body


class LLMClient:
    def __init__(
        self,
        api_key: str,
        tracker: SpendTracker,
        http: httpx.Client | None = None,
        cache: ResponseCache | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set (see .env.example)")
        self.api_key = api_key
        self.tracker = tracker
        self.http = http or httpx.Client(timeout=60)
        self.cache = cache

    def estimate(self, model: str, messages: list[dict[str, str]], max_tokens: int) -> float:
        prompt_tokens = sum(estimate_tokens(m["content"]) for m in messages)
        return PRICES[model].cost(prompt_tokens, max_tokens)

    def is_cached(self, body: dict[str, Any]) -> bool:
        return self.cache is not None and self.cache.get(ResponseCache.key(body)) is not None

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        purpose: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult:
        if model not in PRICES:
            raise ValueError(f"No price for {model}; add it to spend.PRICES before calling it")
        body = request_body(model, messages, max_tokens, temperature, response_format)
        key = ResponseCache.key(body)
        if self.cache is not None and (hit := self.cache.get(key)) is not None:
            return ChatResult(text=hit, cost_usd=0.0, input_tokens=0, output_tokens=0, cached=True)

        call_id = self.tracker.reserve(model, self.estimate(model, messages, max_tokens), purpose)
        # Ask OpenRouter to report the billed cost in the response.
        body = {**body, "usage": {"include": True}}
        resp = self.http.post(
            OPENROUTER_URL,
            json=body,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        # On an HTTP error the reservation stays unsettled and keeps counting (conservative).
        resp.raise_for_status()
        data = resp.json()

        usage = data.get("usage", {})
        input_tokens = int(usage.get("prompt_tokens", 0))
        output_tokens = int(usage.get("completion_tokens", 0))
        cost = usage.get("cost")
        if cost is None:
            cost = PRICES[model].cost(input_tokens, output_tokens)
        self.tracker.settle(
            call_id,
            float(cost),
            {"input_tokens": input_tokens, "output_tokens": output_tokens},
        )
        text = data["choices"][0]["message"]["content"]
        if self.cache is not None:
            self.cache.put(key, text)
        return ChatResult(
            text=text,
            cost_usd=float(cost),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
