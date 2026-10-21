"""Curation steps that need judgment, done with the local LLM (see router/llm.py).

All three steps work from skill text only (names, descriptions, SKILL.md bodies), never from
eval prompts. If a reply can't be parsed, each step falls back to the conservative choice:
keep the skill, keep its description. Parse failures are counted in the run metrics.
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from router.corpus.curate import Removed, Skill, _removed, preference
from router.llm import LLMClient, extract_json

# Ollama serves a few requests in parallel; more workers just queue on a 16 GB laptop.
WORKERS = 2

Request = tuple[list[dict[str, str]], int]  # (messages, max_tokens)

DOMAINS = {
    "software-dev": "writing, reviewing, testing or refactoring code; languages and frameworks",
    "devops-cloud": "deployment, infrastructure, cloud platforms, CI/CD, databases, monitoring",
    "data-analytics": "data analysis, SQL, metrics, machine learning, notebooks",
    "ai-agents": "building or managing AI agents, LLM apps, prompts, skills, agent tooling",
    "design-ui": "UI/UX and visual design, frontend styling, animation, design systems",
    "media-generation": "generating or editing images, video, audio or music",
    "documents-office": "creating or editing documents, PDFs, slides, spreadsheet files",
    "marketing-seo": "marketing, SEO, ads, copywriting, social media, growth",
    "productivity-collab": "email, calendar, chat, workspace apps, notes, task management",
    "business-strategy": "product management, planning, sales, finance, strategy, interviews",
    "research-writing": "research, academic papers, long-form writing, learning",
    "web-automation": "browser control, web scraping, crawling, web search",
}


def _skill_text(s: Skill) -> str:
    return f"name: {s['name']}\ndescription: {s['description']}"


def _system_user(system: str, user: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _call(client: LLMClient, req: Request) -> dict[str, Any] | None:
    messages, max_tokens = req
    return extract_json(client.chat(messages, max_tokens=max_tokens).text)


# --- 1. Pair judging: are two similar skills distinguishable? ---------------------------------

PAIR_SYSTEM = """You review a catalog of AI-agent skills. A router will read only each skill's
name and description and must decide which skills a user's request needs.

Compare skill A and skill B and answer with one verdict:
- "same": they provide essentially the same capability. For almost any request that needs one,
  the other would serve equally well, or the descriptions give no basis for choosing.
- "overlap": different skills, but a meaningful share of requests could be served by either
  (for example, one is a general router or superset of the other).
- "distinct": a typical request clearly needs one rather than the other.

Reply with only a JSON object, no other text:
{"verdict": "same" | "overlap" | "distinct", "reason": "<at most 20 words>"}"""

VERDICTS = {"same", "overlap", "distinct"}
Pair = tuple[float, Skill, Skill]


def pair_request(a: Skill, b: Skill) -> Request:
    return _system_user(PAIR_SYSTEM, f"Skill A\n{_skill_text(a)}\n\nSkill B\n{_skill_text(b)}"), 80


def candidate_pairs(
    skills: list[Skill], vectors: dict[str, np.ndarray], lo: float, hi: float
) -> list[Pair]:
    """Pairs with description cosine in [lo, hi), most similar first."""
    m = np.stack([vectors[s["id"]] for s in skills])
    sims = m @ m.T
    i_idx, j_idx = np.triu_indices(len(skills), 1)
    pairs = [
        (float(sims[i, j]), skills[i], skills[j])
        for i, j in zip(i_idx, j_idx, strict=True)
        if lo <= sims[i, j] < hi
    ]
    return sorted(pairs, key=lambda p: (-p[0], p[1]["id"], p[2]["id"]))


def judge_pairs(client: LLMClient, pairs: list[Pair]) -> list[dict[str, Any]]:
    def judge(pair: Pair) -> dict[str, Any]:
        sim, a, b = pair
        data = _call(client, pair_request(a, b)) or {}
        verdict = data.get("verdict")
        return {
            "a": a["id"],
            "b": b["id"],
            "cosine": round(sim, 4),
            # Unparseable -> treated as distinct, so a bad reply can never delete a skill.
            "verdict": verdict if verdict in VERDICTS else "parse_failed",
            "reason": str(data.get("reason", "")),
        }

    with ThreadPoolExecutor(WORKERS) as pool:
        return list(pool.map(judge, pairs))


def rule_indistinguishable(
    skills: list[Skill], judgments: list[dict[str, Any]]
) -> tuple[list[Skill], list[Removed]]:
    """Drop the less-preferred skill of every 'same' pair, most similar pairs first. A pair is
    skipped if either side was already dropped, so a chain A~B~C can't remove all three."""
    by_id = {s["id"]: s for s in skills}
    dropped: dict[str, Removed] = {}
    for j in judgments:
        if j["verdict"] != "same" or j["a"] in dropped or j["b"] in dropped:
            continue
        a, b = by_id[j["a"]], by_id[j["b"]]
        keep, drop = (a, b) if preference(a) >= preference(b) else (b, a)
        dropped[drop["id"]] = _removed(
            drop, "indistinguishable", kept_id=keep["id"], cosine=j["cosine"], reason=j["reason"]
        )
    return [s for s in skills if s["id"] not in dropped], list(dropped.values())


# --- 2. Classification: domain, and whether the description says what the skill does ---------

CLASSIFY_SYSTEM = f"""You classify an AI-agent skill from its name and description.

1. domain: the single best fit from this list:
{chr(10).join(f"   - {k}: {v}" for k, v in DOMAINS.items())}
2. states_what: true if the description says what the skill does or produces (its capability
   or output). false if it only says when to trigger it (e.g. a list of phrases a user might
   say) without saying what the skill actually does.

Reply with only a JSON object, no other text:
{{"domain": "<one key from the list>", "states_what": true | false}}"""


def classify_request(s: Skill) -> Request:
    return _system_user(CLASSIFY_SYSTEM, _skill_text(s)), 40


def classify(client: LLMClient, skills: list[Skill]) -> dict[str, dict[str, Any]]:
    def one(s: Skill) -> tuple[str, dict[str, Any]]:
        data = _call(client, classify_request(s))
        if data is None:
            return s["id"], {"domain": "parse_failed", "states_what": True}
        domain = data.get("domain")
        return s["id"], {
            "domain": domain if domain in DOMAINS else "unclassified",
            "states_what": data.get("states_what") is not False,
        }

    with ThreadPoolExecutor(WORKERS) as pool:
        return dict(pool.map(one, skills))


# --- 3. Rewrite trigger-only descriptions ------------------------------------------------------

REWRITE_SYSTEM = """You rewrite the description of an AI-agent skill so a router can tell what
the skill does. Use only facts from the SKILL.md provided; do not invent features.

Write at most 60 words: first what the skill does or produces, then one short clause on when to
use it. Plain text, no quotes, no markdown.

Reply with only a JSON object, no other text: {"description": "<new description>"}"""

BODY_CHARS = 4000  # enough of SKILL.md to see what the skill does, and fast to process locally


def rewrite_request(s: Skill, body: str) -> Request:
    user = f"{_skill_text(s)}\n\nSKILL.md body:\n{body[:BODY_CHARS]}"
    return _system_user(REWRITE_SYSTEM, user), 150


def rewrite(
    client: LLMClient, skills: list[Skill], read_body: Callable[[Skill], str]
) -> dict[str, str | None]:
    """Return id -> new description, or None where the reply couldn't be parsed."""

    def one(s: Skill) -> tuple[str, str | None]:
        data = _call(client, rewrite_request(s, read_body(s)))
        text = (
            " ".join(str(data["description"]).split()) if data and data.get("description") else ""
        )
        return s["id"], text or None

    with ThreadPoolExecutor(WORKERS) as pool:
        return dict(pool.map(one, skills))
