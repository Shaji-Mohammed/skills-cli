"""Eval metrics, written before any router exists (ROADMAP Phase 2).

Definitions (see evals/LABELING.md):
- Exact-set match, strict: the returned set equals the gold set.
- Exact-set match, equivalence-aware: same after mapping every skill to its equivalence set;
  returning two members of one set where gold needs one is over-routing, not a match.
- An empty gold set (no-skill prompt) matches only an empty prediction.
"""

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from router.stats import wilson

# skill id -> equivalence-set id. Skills not in any set map to themselves.
Equivalence = Mapping[str, str]

FAILURES = ("MISS_STAGE1", "OVERROUTING", "WRONG_SCORER", "ABSTAIN")


@dataclass(frozen=True)
class Item:
    id: str
    bucket: str  # explicit | implied | no_skill
    source: str  # model_drafted | hand_written
    gold: frozenset[str]

    @property
    def multi_skill(self) -> bool:
        return len(self.gold) >= 2


@dataclass(frozen=True)
class Prediction:
    id: str
    skills: tuple[str, ...]
    shortlist: tuple[str, ...] | None = None  # Stage 1 output, if the router reports it
    abstained: bool = False
    path: str = "default"  # e.g. fast | scorer | fallback
    latency_ms: float = 0.0


def _canon(skills: Iterable[str], eq: Equivalence) -> list[str]:
    return [eq.get(s, s) for s in skills]


def exact_strict(pred: Iterable[str], gold: frozenset[str]) -> bool:
    return set(pred) == gold


def exact_equivalence(pred: Iterable[str], gold: frozenset[str], eq: Equivalence) -> bool:
    mapped = _canon(pred, eq)
    if len(mapped) != len(set(mapped)):
        return False  # two members of one set returned: over-routing
    return set(mapped) == set(_canon(gold, eq))


def precision_recall(
    pred: Iterable[str], gold: frozenset[str], eq: Equivalence
) -> tuple[float, float]:
    """Per-prompt precision and recall on equivalence-mapped skills.

    Empty prediction: precision 1 (nothing wrong was returned). Empty gold: recall 1 (nothing
    could be missed). So a no-skill prompt answered with skills gets precision 0, recall 1.
    """
    p = _canon(pred, eq)
    g = set(_canon(gold, eq))
    hits = len(set(p) & g)
    precision = hits / len(p) if p else 1.0
    recall = hits / len(g) if g else 1.0
    return precision, recall


def stage1_recall(shortlist: Iterable[str], gold: frozenset[str], eq: Equivalence) -> bool:
    """All gold skills (or an equivalent) are in the shortlist. Vacuously true for empty gold."""
    return set(_canon(gold, eq)) <= set(_canon(shortlist, eq))


def failure_type(item: Item, pred: Prediction, eq: Equivalence) -> str | None:
    """None for an equivalence-aware exact match, otherwise one of FAILURES."""
    if exact_equivalence(pred.skills, item.gold, eq):
        return None
    if pred.abstained and item.gold:
        return "ABSTAIN"
    if pred.shortlist is not None and not stage1_recall(pred.shortlist, item.gold, eq):
        return "MISS_STAGE1"
    mapped = _canon(pred.skills, eq)
    gold = set(_canon(item.gold, eq))
    if gold <= set(mapped):
        return "OVERROUTING"  # all gold present, plus extras or duplicate equivalents
    return "WRONG_SCORER"


def _rate(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson(k, n)
    return {
        "k": k,
        "n": n,
        "rate": round(k / n, 4) if n else None,
        "wilson95": [round(lo, 4), round(hi, 4)],
    }


@dataclass
class _Tally:
    strict: int = 0
    equivalence: int = 0
    stage1: int = 0
    stage1_n: int = 0
    n: int = 0
    precision: list[float] = field(default_factory=list)
    recall: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "exact_strict": _rate(self.strict, self.n),
            "exact_equivalence": _rate(self.equivalence, self.n),
            "stage1_recall": _rate(self.stage1, self.stage1_n) if self.stage1_n else None,
            "mean_precision": round(float(np.mean(self.precision)), 4) if self.n else None,
            "mean_recall": round(float(np.mean(self.recall)), 4) if self.n else None,
        }


def summarize(items: list[Item], predictions: list[Prediction], eq: Equivalence) -> dict[str, Any]:
    """Overall, per-bucket, multi-skill and per-source metrics with 95% Wilson intervals."""
    by_id = {p.id: p for p in predictions}
    missing = [i.id for i in items if i.id not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} items have no prediction, e.g. {missing[:3]}")

    groups: dict[str, _Tally] = defaultdict(_Tally)
    failures: Counter[str] = Counter()
    latency: dict[str, list[float]] = defaultdict(list)
    for item in items:
        pred = by_id[item.id]
        keys = ["overall", f"bucket:{item.bucket}", f"source:{item.source}"]
        if item.multi_skill:
            keys.append("multi_skill")
        strict = exact_strict(pred.skills, item.gold)
        equiv = exact_equivalence(pred.skills, item.gold, eq)
        precision, recall = precision_recall(pred.skills, item.gold, eq)
        for key in keys:
            t = groups[key]
            t.n += 1
            t.strict += strict
            t.equivalence += equiv
            t.precision.append(precision)
            t.recall.append(recall)
            if pred.shortlist is not None:
                t.stage1_n += 1
                t.stage1 += stage1_recall(pred.shortlist, item.gold, eq)
        if (f := failure_type(item, pred, eq)) is not None:
            failures[f] += 1
        latency[pred.path].append(pred.latency_ms)

    n = len(items)
    return {
        "groups": {k: groups[k].summary() for k in sorted(groups)},
        "failures": {f: failures[f] for f in FAILURES},
        "paths": {
            path: {
                "share": _rate(len(v), n),
                "latency_ms_p50": round(float(np.percentile(v, 50)), 1),
                "latency_ms_p95": round(float(np.percentile(v, 95)), 1),
            }
            for path, v in sorted(latency.items())
        },
    }
