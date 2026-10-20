"""OpenRouter chat client: the only module that makes paid calls, all through SpendTracker."""

from dataclasses import dataclass
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


class LLMClient:
    def __init__(
        self,
        api_key: str,
        tracker: SpendTracker,
        http: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is not set (see .env.example)")
        self.api_key = api_key
        self.tracker = tracker
        self.http = http or httpx.Client(timeout=60)

    def estimate(self, model: str, messages: list[dict[str, str]], max_tokens: int) -> float:
        prompt_tokens = sum(estimate_tokens(m["content"]) for m in messages)
        return PRICES[model].cost(prompt_tokens, max_tokens)

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
        call_id = self.tracker.reserve(model, self.estimate(model, messages, max_tokens), purpose)

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            # Ask OpenRouter to report the billed cost in the response.
            "usage": {"include": True},
        }
        if response_format is not None:
            body["response_format"] = response_format
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
        return ChatResult(
            text=data["choices"][0]["message"]["content"],
            cost_usd=float(cost),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
