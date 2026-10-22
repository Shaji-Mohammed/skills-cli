"""Phase 3 evaluation on dev: Stage 1 recall sweeps and the end-to-end retrieval baseline.

Only the dev split is loaded here (router.splits keeps test locked). Every run is saved under
results/ with its config, corpus and eval-set versions and git commit.
"""

import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from router.corpus.descriptions import DescriptionSource
from router.evalset.equivalence import EQUIVALENCE, SKILLS, load_equivalence
from router.retrieval import HybridIndex, RetrievalConfig, Shortlist
from router.runs import start_run
from router.score import Item, Prediction, stage1_recall, summarize
from router.splits import load_split, split_path
from router.stats import wilson

STAGE1_CAPS = (10, 20, 40)
MODELS = ("BAAI/bge-small-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2")
WEIGHTS = (0.5, 0.6, 0.7, 0.8, 0.9)
SOURCES: tuple[DescriptionSource, ...] = ("original", "rewritten")


@dataclass(frozen=True)
class BaselineConfig:
    """Answer = the top `answer_k` shortlist skills scoring >= answer_gate * top score;
    abstain (return nothing) when the best raw cosine is below abstain_cosine."""

    retrieval: RetrievalConfig
    answer_k: int = 1
    answer_gate: float = 0.9
    abstain_cosine: float = 0.5


def load_items(split: str) -> list[Item]:
    return [
        Item(id=r["id"], bucket=r["bucket"], source=r["source"], gold=frozenset(r["gold"]))
        for r in load_split(split)
    ]


def load_skills() -> list[dict[str, Any]]:
    return [json.loads(line) for line in SKILLS.read_text().splitlines() if line.strip()]


def _rate(k: int, n: int) -> dict[str, Any]:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 4), "wilson95": [round(lo, 4), round(hi, 4)]}


def stage1_metrics(
    items: list[Item], shortlists: list[Shortlist], cap: int, eq: dict[str, str]
) -> dict[str, Any]:
    """Recall at each cap (equivalence-aware) over items with a non-empty gold set, plus
    candidate counts and latency at the configured cap."""
    with_gold = [(it, sl) for it, sl in zip(items, shortlists, strict=True) if it.gold]
    out: dict[str, Any] = {}
    for c in STAGE1_CAPS:
        hits = sum(stage1_recall(sl.ids[:c], it.gold, eq) for it, sl in with_gold)
        out[f"recall@{c}"] = _rate(hits, len(with_gold))
        implied = [(it, sl) for it, sl in with_gold if it.bucket == "implied"]
        out[f"implied_recall@{c}"] = _rate(
            sum(stage1_recall(sl.ids[:c], it.gold, eq) for it, sl in implied), len(implied)
        )
    lat = [sl.latency_ms for sl in shortlists]
    out["median_candidates"] = median(min(len(sl.ids), cap) for sl in shortlists)
    out["latency_ms_p50"] = round(float(np.percentile(lat, 50)), 2)
    out["latency_ms_p95"] = round(float(np.percentile(lat, 95)), 2)
    return out


def run_stage1_sweep(allow_dirty: bool = False) -> Path:
    """Every model x weight x description source on dev, gate 0 so recall@cap is pure rank."""
    skills, eq, items = load_skills(), load_equivalence(EQUIVALENCE), load_items("dev")
    configs = [
        RetrievalConfig(
            embedding_model=m,
            description_source=src,
            embedding_weight=w,
            gate=0.0,
            cap=max(STAGE1_CAPS),
        )
        for m, src, w in itertools.product(MODELS, SOURCES, (*WEIGHTS, 0.0, 1.0))
    ]
    run = start_run(
        "stage1_sweep",
        config={"configs": [asdict(c) for c in configs], "caps": STAGE1_CAPS},
        corpus_path=SKILLS,
        evalset_path=split_path("dev"),
        allow_dirty=allow_dirty,
    )
    results = []
    for cfg in configs:
        index = HybridIndex(skills, cfg)
        shortlists = [index.shortlist(it_prompt) for it_prompt in _prompts("dev")]
        results.append({"config": asdict(cfg), **stage1_metrics(items, shortlists, cfg.cap, eq)})
        print(
            f"{cfg.embedding_model.split('/')[-1]:24s} {cfg.description_source:9s} "
            f"w={cfg.embedding_weight:.1f} recall@20={results[-1]['recall@20']['rate']}"
        )
    best = max(results, key=lambda r: (r["recall@20"]["rate"], r["recall@40"]["rate"]))
    run.write_metrics({"results": results, "best_by_recall@20": best["config"]})
    return run.path


def _prompts(split: str) -> list[str]:
    return [r["prompt"] for r in load_split(split)]


def predict(index: HybridIndex, prompt: str, item_id: str, cfg: BaselineConfig) -> Prediction:
    return answer(index.shortlist(prompt), item_id, cfg)


def run_baseline(cfg: BaselineConfig, allow_dirty: bool = False) -> Path:
    skills, eq, items = load_skills(), load_equivalence(EQUIVALENCE), load_items("dev")
    run = start_run(
        "baseline",
        config={**asdict(cfg)},
        corpus_path=SKILLS,
        evalset_path=split_path("dev"),
        allow_dirty=allow_dirty,
    )
    index = HybridIndex(skills, cfg.retrieval)
    prompts = _prompts("dev")
    preds = [predict(index, p, it.id, cfg) for p, it in zip(prompts, items, strict=True)]
    run.write_predictions(
        {
            "id": p.id,
            "skills": list(p.skills),
            "shortlist": list(p.shortlist or ()),
            "abstained": p.abstained,
            "latency_ms": round(p.latency_ms, 2),
        }
        for p in preds
    )
    run.write_metrics(summarize(items, preds, eq))
    return run.path


def sweep_baseline(retrieval: RetrievalConfig, allow_dirty: bool = False) -> Path:
    """Grid over answer_k, answer_gate and abstain_cosine for a fixed retriever, on dev."""
    skills, eq, items = load_skills(), load_equivalence(EQUIVALENCE), load_items("dev")
    grid = [
        BaselineConfig(retrieval, k, g, a)
        for k, g, a in itertools.product(
            (1, 2, 3), (0.8, 0.85, 0.9, 0.95), (0.4, 0.5, 0.55, 0.6, 0.65)
        )
    ]
    run = start_run(
        "baseline_sweep",
        config={"retrieval": asdict(retrieval), "grid_size": len(grid)},
        corpus_path=SKILLS,
        evalset_path=split_path("dev"),
        allow_dirty=allow_dirty,
    )
    index = HybridIndex(skills, retrieval)
    prompts = _prompts("dev")
    shortlists = [index.shortlist(p) for p in prompts]
    rows = []
    for cfg in grid:
        preds = [answer(sl, it.id, cfg) for sl, it in zip(shortlists, items, strict=True)]
        m = summarize(items, preds, eq)
        rows.append(
            {
                "answer_k": cfg.answer_k,
                "answer_gate": cfg.answer_gate,
                "abstain_cosine": cfg.abstain_cosine,
                "exact_equivalence": m["groups"]["overall"]["exact_equivalence"],
                "exact_strict": m["groups"]["overall"]["exact_strict"],
            }
        )
    best = max(rows, key=lambda r: r["exact_equivalence"]["rate"])
    run.write_metrics({"grid": rows, "best": best})
    return run.path


def answer(sl: Shortlist, item_id: str, cfg: BaselineConfig) -> Prediction:
    """Turn a shortlist into the baseline's answer. Separate from retrieval so a grid over
    answer settings reuses one set of shortlists."""
    if not sl.ids or sl.top_cosine < cfg.abstain_cosine:
        return Prediction(
            item_id, (), shortlist=sl.ids, abstained=True, path="abstain", latency_ms=sl.latency_ms
        )
    floor = cfg.answer_gate * sl.scores[0]
    picked = tuple(i for i, s in zip(sl.ids, sl.scores, strict=True) if s >= floor)
    return Prediction(
        item_id,
        picked[: cfg.answer_k],
        shortlist=sl.ids,
        path="retrieval",
        latency_ms=sl.latency_ms,
    )


STAGE1_CONFIG = SKILLS.parent.parent / "configs" / "stage1.json"


def stage1_check(path: Path = STAGE1_CONFIG) -> tuple[dict[str, Any], bool]:
    """CI regression check: dev recall@20 for the pinned Stage 1 config must stay above its
    floor. Deterministic (no LLM calls) and doesn't write a results run."""
    spec = json.loads(path.read_text())
    cfg = RetrievalConfig(**spec["retrieval"])
    skills, eq, items = load_skills(), load_equivalence(EQUIVALENCE), load_items("dev")
    index = HybridIndex(skills, cfg)
    shortlists = [index.shortlist(p, cap=max(STAGE1_CAPS)) for p in _prompts("dev")]
    metrics = stage1_metrics(items, shortlists, cfg.cap, eq)
    return metrics, metrics["recall@20"]["rate"] >= spec["min_recall_at_20"]
