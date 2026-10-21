"""Local LLM client (Ollama). Free to call, so it doesn't go through the spend tracker."""

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass(frozen=True)
class ChatResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cached: bool = False


class ResponseCache:
    """Append-only JSONL of request-hash -> response text.

    Committed alongside the data it produced, so rebuilding the corpus replays the same LLM
    outputs instead of re-running the model (and getting slightly different answers).
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


def extract_json(text: str) -> dict[str, Any] | None:
    """Parse the first JSON object in a reply. qwen3.5 on Ollama has no structured-output mode,
    so replies can arrive wrapped in code fences or with stray text around the object."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


class LLMClient:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        http: httpx.Client | None = None,
        cache: ResponseCache | None = None,
    ) -> None:
        self.model = model
        self.url = f"{base_url.rstrip('/')}/api/chat"
        self.http = http or httpx.Client(timeout=300)
        self.cache = cache

    def chat(self, messages: list[dict[str, str]], max_tokens: int = 256) -> ChatResult:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            # qwen3.5 thinks by default (~25 s for a one-line reply); these are short judgments.
            "think": False,
            "options": {"temperature": 0, "num_predict": max_tokens},
        }
        key = ResponseCache.key(body)
        if self.cache is not None and (hit := self.cache.get(key)) is not None:
            return ChatResult(text=hit, input_tokens=0, output_tokens=0, latency_s=0.0, cached=True)

        start = time.perf_counter()
        resp = self.http.post(self.url, json=body)
        resp.raise_for_status()
        data = resp.json()
        latency = time.perf_counter() - start

        text = data["message"]["content"]
        if self.cache is not None:
            self.cache.put(key, text)
        return ChatResult(
            text=text,
            input_tokens=int(data.get("prompt_eval_count", 0)),
            output_tokens=int(data.get("eval_count", 0)),
            latency_s=latency,
        )
