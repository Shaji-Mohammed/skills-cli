"""qwen-drafted eval prompts: 200 explicit, 95 implied, 80 no-skill (evals/LABELING.md).

Drafts are written from SKILL.md bodies only, never from descriptions. Gold labels are always
canonical ids. Product-locked skills are explicit-only. Every draft passes D1-D3 here (D4 runs
on the pooled set). Slots are oversampled, so filter drops don't leave a bucket short.
"""

import csv
import json
import math
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from router.config import REPO_ROOT
from router.corpus.curate_llm import WORKERS, _system_user
from router.corpus.pipeline import read_body
from router.corpus.product_locked import FROZEN as PRODUCT_LOCKED
from router.corpus.product_locked import load_product_locked
from router.evalset.equivalence import EQUIVALENCE, SKILLS, load_equivalence
from router.evalset.filters import canonical_gold, check_prompt, members_of
from router.evalset.handwritten import MAX_GOLD_SETS_PER_SKILL
from router.evalset.slots import SHEET, cooccur_request, cross_reference_pairs
from router.llm import LLMClient, extract_json
from router.runs import file_sha256, start_run

OUT = REPO_ROOT / "evals" / "drafts" / "drafts.jsonl"
DROPPED = REPO_ROOT / "evals" / "drafts" / "dropped.jsonl"
SEED = 20261023
TEMPERATURE = 0.7
OVERSAMPLE = 1.15

TARGETS: dict[tuple[str, str], int] = {
    ("explicit", "single"): 130,
    ("explicit", "pair"): 52,
    ("explicit", "triple"): 18,
    ("implied", "single"): 70,
    ("implied", "pair"): 25,
    ("no_skill", "none"): 80,
}
BODY_CHARS = {1: 2500, 2: 1800, 3: 1300}  # per skill; keeps local generation time bounded

NO_SKILL_TOPICS = [
    "a general programming concept",
    "a math question",
    "a history question",
    "cooking advice",
    "travel planning advice",
    "a personal finance concept",
    "a grammar or word-usage question",
    "trivia",
    "general health information",
    "a unit conversion",
    "casual small talk",
    "explaining a joke or idiom",
    "a philosophy question",
    "career advice",
    "a book or film recommendation",
    "a science concept",
    "an everyday errand outside software (booking, buying, calling someone)",
    "a riddle or logic puzzle",
    "relationship or etiquette advice",
    "a quick factual comparison of two everyday things",
]

Skill = dict[str, Any]
Slot = dict[str, Any]

TRIPLE_SYSTEM = """You judge whether three AI-agent skills are used together.

Answer true only if one realistic user task would REQUIRE all three skills, each supplying a
distinct part of the work. Answer false if any one is optional, an alternative to another, or
a general version of another.

Reply with only a JSON object, no other text:
{"cooccur": true | false, "reason": "<at most 20 words>"}"""

DRAFT_RULES = {
    "explicit": "The request must name the file format, tool, platform, product or service "
    "involved, the way a real user would.",
    "implied": "The request must describe only the goal or situation. It may mention what the "
    "user already has (their app, cloud account, workspace or framework), but it must not name "
    "the file format, tool, product or service that would do the job.",
}

DRAFT_SYSTEM = """You write one realistic request that a user might send to an AI assistant.

{need}
{bucket_rule}
If a skill is a working method (debugging, test-first development, planning, reviewing), the
request must explicitly ask for that method, described in plain words.
Write in the user's own words, 1 to 3 sentences. Don't copy phrases from the skill text, and
don't mention skills or any skill's name.

Reply with only a JSON object, no other text: {{"prompt": "<the request>"}}"""

NO_SKILL_SYSTEM = """You write one realistic message that a user might send to an AI assistant.

It must need no specialized tool, file format, platform or integration: it can be answered from
general knowledge or simple reasoning, or it asks for something outside software altogether.
Topic: {topic}. Write in the user's own words, 1 to 3 sentences.

Reply with only a JSON object, no other text: {{"prompt": "<the message>"}}"""


def triple_request(skills: list[Skill]) -> tuple[list[dict[str, str]], int]:
    user = "\n\n".join(
        f"Skill {'ABC'[i]}\nname: {s['name']}\ndescription: {s['description_original']}"
        for i, s in enumerate(skills)
    )
    return _system_user(TRIPLE_SYSTEM, user), 60


def draft_request(slot: Slot, by_id: dict[str, Skill]) -> tuple[list[dict[str, str]], int]:
    if slot["bucket"] == "no_skill":
        return _system_user(NO_SKILL_SYSTEM.format(topic=slot["topic"]), "Write the message."), 120
    gold = slot["drafted_from"]
    need = (
        "The request must need the skill below to be done to spec."
        if len(gold) == 1
        else f"The request must need all {len(gold)} skills below, each doing a distinct part."
    )
    system = DRAFT_SYSTEM.format(need=need, bucket_rule=DRAFT_RULES[slot["bucket"]])
    chars = BODY_CHARS[len(gold)]
    user = "\n\n".join(
        f"Skill {i + 1} (SKILL.md excerpt):\n{read_body(by_id[g])[:chars]}"
        for i, g in enumerate(gold)
    )
    return _system_user(system, user), 160


class _Planner:
    """Assigns gold sets to slots, enforcing the per-skill cap across hand-written prompts and
    drafts. Usage counts include oversampled slots, so the cap holds whatever is trimmed."""

    def __init__(self, skills: list[Skill], eq: dict[str, str], locked: set[str]) -> None:
        self.by_id = {s["id"]: s for s in skills}
        self.eq = eq
        self.locked = locked
        self.canonical = sorted({canonical_gold([s["id"]], eq)[0] for s in skills})
        self.usage: Counter[str] = Counter()

    def fits(self, ids: list[str]) -> bool:
        return all(self.usage[i] < MAX_GOLD_SETS_PER_SKILL for i in ids)

    def take(self, ids: list[str]) -> None:
        self.usage.update(ids)


def _hand_written_usage(planner: _Planner) -> set[tuple[str, str]]:
    """Count selected hand-written slots toward the cap; return every pair already in the
    sheet so drafts don't repeat a hand-written combination."""
    with SHEET.open(newline="") as f:
        rows = list(csv.DictReader(f))
    pairs: set[tuple[str, str]] = set()
    for r in rows:
        ids = r["gold"].split(";")
        if r.get("use") == "yes":
            planner.take(ids)
        if r["kind"] == "pair":
            a, b = sorted(ids)
            pairs.add((a, b))
    return pairs


def make_slot(
    n: int, bucket: str, kind: str, drafted_from: list[str], eq: dict[str, str], **extra: Any
) -> Slot:
    """A draft slot. Gold is always canonical, even when the text comes from a non-canonical
    member's SKILL.md."""
    return {
        "slot": f"{bucket}-{kind}-{n}",
        "bucket": bucket,
        "kind": kind,
        "drafted_from": drafted_from,
        "gold": canonical_gold(drafted_from, eq),
        **extra,
    }


def plan(
    client: LLMClient, skills: list[Skill], eq: dict[str, str], locked: set[str], rng: random.Random
) -> tuple[list[Slot], dict[str, Any]]:
    planner = _Planner(skills, eq, locked)
    seen_pairs = _hand_written_usage(planner)
    want = {k: math.ceil(n * OVERSAMPLE) for k, n in TARGETS.items()}
    slots: list[Slot] = []
    stats: Counter[str] = Counter()

    def add(bucket: str, kind: str, gold: list[str], **extra: Any) -> None:
        slots.append(make_slot(len(slots) + 1, bucket, kind, gold, eq, **extra))
        planner.take(gold)

    # Pairs: judged in shuffled order; implied pairs only from non-locked skills.
    pool = [p for p in cross_reference_pairs(skills, eq) if p not in seen_pairs]
    rng.shuffle(pool)
    accepted: list[tuple[str, str]] = []
    need_pairs = {"implied": want[("implied", "pair")], "explicit": want[("explicit", "pair")]}
    for a, b in pool:
        if not any(need_pairs.values()):
            break
        if not planner.fits([a, b]):
            continue
        bucket = "implied" if need_pairs["implied"] and not {a, b} & locked else "explicit"
        if not need_pairs[bucket]:
            continue
        reply = client.chat(*cooccur_request(planner.by_id[a], planner.by_id[b]))
        data = extract_json(reply.text) or {}
        stats["pairs_judged"] += 1
        if data.get("cooccur") is True:
            add(bucket, "pair", [a, b], pair_reason=str(data.get("reason", "")))
            accepted.append((a, b))
            need_pairs[bucket] -= 1

    # Triples: extend an accepted pair with a third skill cross-referenced by either member.
    neighbours: dict[str, set[str]] = {}
    for a, b in pool:
        neighbours.setdefault(a, set()).add(b)
        neighbours.setdefault(b, set()).add(a)
    need_triples = want[("explicit", "triple")]
    tried: set[frozenset[str]] = set()
    for a, b in accepted:
        for c in sorted((neighbours.get(a, set()) | neighbours.get(b, set())) - {a, b}):
            if not need_triples:
                break
            key = frozenset((a, b, c))
            if key in tried or not planner.fits([a, b, c]):
                continue
            tried.add(key)
            trio = [planner.by_id[x] for x in (a, b, c)]
            data = extract_json(client.chat(*triple_request(trio)).text) or {}
            stats["triples_judged"] += 1
            if data.get("cooccur") is True:
                add("explicit", "triple", sorted([a, b, c]))
                need_triples -= 1

    # Singles: least-used skills first (spreads coverage), random among ties.
    for bucket in ("implied", "explicit"):
        n = want[(bucket, "single")]
        candidates = [c for c in planner.canonical if bucket == "explicit" or c not in locked]
        order = sorted(candidates, key=lambda c: (planner.usage[c], rng.random()))
        for c in order:
            if not n:
                break
            if planner.fits([c]):
                add(bucket, "single", [c])
                n -= 1

    for i in range(want[("no_skill", "none")]):
        topic = NO_SKILL_TOPICS[i % len(NO_SKILL_TOPICS)]
        slots.append(make_slot(len(slots) + 1, "no_skill", "none", [], eq, topic=topic))
    stats["slots_planned"] = len(slots)
    return slots, dict(stats)


def draft_all(
    client: LLMClient, slots: list[Slot], by_id: dict[str, Skill], eq: dict[str, str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Draft each slot; D1-D3 must pass, with one retry on a new seed."""

    def one(item: tuple[int, Slot]) -> dict[str, Any]:
        i, slot = item
        gold_skills = [by_id[m] for g in slot["gold"] for m in members_of(g, eq)]
        problems: list[str] = []
        for attempt in range(2):
            seed = SEED + i + 100_000 * attempt
            messages, max_tokens = draft_request(slot, by_id)
            reply = client.chat(messages, max_tokens=max_tokens, temperature=TEMPERATURE, seed=seed)
            data = extract_json(reply.text)
            prompt = " ".join(str(data.get("prompt", "")).split()) if data else ""
            problems = ["parse_failed"] if not prompt else check_prompt(prompt, gold_skills)
            if not problems:
                return {**slot, "prompt": prompt, "seed": seed, "ok": True}
        return {**slot, "prompt": prompt, "seed": seed, "ok": False, "problems": problems}

    with ThreadPoolExecutor(WORKERS) as pool:
        results = list(pool.map(one, enumerate(slots)))
    return [r for r in results if r["ok"]], [r for r in results if not r["ok"]]


def trim(drafted: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the first TARGETS[k] passing drafts per stratum, in slot order."""
    kept: list[dict[str, Any]] = []
    for key, n in TARGETS.items():
        kept += [d for d in drafted if (d["bucket"], d["kind"]) == key][:n]
    return kept


def generate(client: LLMClient, allow_dirty: bool = False) -> Path:
    skills = [json.loads(line) for line in SKILLS.read_text().splitlines() if line.strip()]
    eq = load_equivalence(EQUIVALENCE)
    locked = load_product_locked()
    if not locked:
        raise ValueError("Freeze corpus/product_locked.json first")
    by_id = {s["id"]: s for s in skills}
    run = start_run(
        "drafts",
        config={
            "seed": SEED,
            "temperature": TEMPERATURE,
            "oversample": OVERSAMPLE,
            "targets": {f"{b}/{k}": n for (b, k), n in TARGETS.items()},
            "body_chars": BODY_CHARS,
            "draft_system": DRAFT_SYSTEM,
            "draft_rules": DRAFT_RULES,
            "no_skill_system": NO_SKILL_SYSTEM,
            "no_skill_topics": NO_SKILL_TOPICS,
            "triple_system": TRIPLE_SYSTEM,
            "max_gold_sets_per_skill": MAX_GOLD_SETS_PER_SKILL,
            "llm_model": client.model,
            "equivalence_sha256": file_sha256(EQUIVALENCE),
            "product_locked_sha256": file_sha256(PRODUCT_LOCKED),
            "handwritten_sheet_sha256": file_sha256(SHEET),
        },
        corpus_path=SKILLS,
        evalset_path=None,
        allow_dirty=allow_dirty,
    )
    rng = random.Random(SEED)
    slots, stats = plan(client, skills, eq, locked, rng)
    print(f"Planned {len(slots)} slots; drafting with {client.model}")
    drafted, failed = draft_all(client, slots, by_id, eq)
    kept = trim(drafted)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = ("slot", "bucket", "kind", "gold", "drafted_from", "prompt", "seed")
    OUT.write_text(
        "".join(
            json.dumps(
                {
                    **{k: d[k] for k in fields},
                    "source": "model_drafted",
                    "multi_skill": len(d["gold"]) >= 2,
                },
                ensure_ascii=False,
            )
            + "\n"
            for d in kept
        )
    )
    DROPPED.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in failed))
    counts = Counter(f"{d['bucket']}/{d['kind']}" for d in kept)
    run.write_metrics(
        {
            **stats,
            "drafted_ok": len(drafted),
            "dropped_by_filters": len(failed),
            "drop_reasons": dict(Counter(p.split(":")[0] for d in failed for p in d["problems"])),
            "kept": len(kept),
            "kept_by_stratum": dict(counts),
            "short_strata": {
                f"{b}/{k}": n - counts[f"{b}/{k}"]
                for (b, k), n in TARGETS.items()
                if counts[f"{b}/{k}"] < n
            },
            "distinct_gold_skills": len({g for d in kept for g in d["gold"]}),
            "drafts_sha256": file_sha256(OUT),
        }
    )
    return run.path
