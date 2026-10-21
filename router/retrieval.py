"""Stage 1: hybrid retrieval (local embeddings + BM25) with a relative gate and a hard cap.

Scores per query:
    emb  = cosine(prompt, skill text), min-max scaled over the corpus for this query
    bm25 = BM25 score / the query's top BM25 score (0 if no term matches)
    hybrid = w * emb + (1 - w) * bm25
The shortlist keeps skills with hybrid >= gate * top hybrid, at most `cap`. Raw top cosine is
kept separately as an absolute signal, since min-max scaling hides how good the best match is.
"""

import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from router.corpus.descriptions import DescriptionSource, description_for
from router.embed import embed


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass(frozen=True)
class RetrievalConfig:
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    description_source: DescriptionSource = "original"
    embedding_weight: float = 0.7
    gate: float = 0.6  # keep candidates scoring at least this fraction of the top score
    cap: int = 20


@dataclass(frozen=True)
class Shortlist:
    ids: tuple[str, ...]
    scores: tuple[float, ...]
    top_cosine: float  # absolute similarity of the best match, before scaling
    latency_ms: float


class HybridIndex:
    def __init__(self, skills: list[dict[str, Any]], config: RetrievalConfig) -> None:
        self.config = config
        self.ids = [s["id"] for s in skills]
        texts = [f"{s['name']}: {description_for(s, config.description_source)}" for s in skills]
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])
        self.vectors = embed(texts, model=config.embedding_model)

    def scores(self, prompt: str) -> tuple[np.ndarray, float]:
        """(hybrid score per skill, raw top cosine)."""
        cos = self.vectors @ embed([prompt], model=self.config.embedding_model)[0]
        span = float(cos.max() - cos.min()) or 1.0
        emb = (cos - cos.min()) / span
        lexical = self.bm25.get_scores(tokenize(prompt))
        top_lexical = float(lexical.max())
        bm25 = lexical / top_lexical if top_lexical > 0 else np.zeros_like(lexical)
        w = self.config.embedding_weight
        return w * emb + (1 - w) * bm25, float(cos.max())

    def shortlist(self, prompt: str, cap: int | None = None) -> Shortlist:
        start = time.perf_counter()
        hybrid, top_cos = self.scores(prompt)
        order = np.argsort(-hybrid)
        floor = self.config.gate * float(hybrid[order[0]])
        keep = [int(i) for i in order[: cap or self.config.cap] if hybrid[i] >= floor]
        return Shortlist(
            ids=tuple(self.ids[i] for i in keep),
            scores=tuple(round(float(hybrid[i]), 4) for i in keep),
            top_cosine=round(top_cos, 4),
            latency_ms=(time.perf_counter() - start) * 1000,
        )
