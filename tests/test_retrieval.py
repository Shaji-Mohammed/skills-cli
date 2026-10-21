from typing import Any

import numpy as np
import pytest

from router import retrieval
from router.retrieval import HybridIndex, RetrievalConfig, tokenize

SKILLS = [
    {
        "id": "a/pdf",
        "name": "pdf",
        "description_original": "merge split and edit pdf files",
        "description": "merge split and edit pdf files",
    },
    {
        "id": "a/xlsx",
        "name": "xlsx",
        "description_original": "spreadsheet formulas and charts",
        "description": "build spreadsheets with formulas and charts",
    },
    {
        "id": "a/seo",
        "name": "seo",
        "description_original": "audit search rankings",
        "description": "audit search rankings",
    },
]


def fake_embed(texts: list[str], model: str = "") -> np.ndarray:
    """Deterministic bag-of-words vectors over a tiny vocabulary, L2-normalized."""
    vocab = ["pdf", "spreadsheet", "spreadsheets", "search", "merge", "charts"]
    v = np.array([[float(w in tokenize(t)) for w in vocab] + [0.01] for t in texts])
    normalized: np.ndarray = v / np.linalg.norm(v, axis=1, keepdims=True)
    return normalized


@pytest.fixture(autouse=True)
def no_model(monkeypatch: Any) -> None:
    monkeypatch.setattr(retrieval, "embed", fake_embed)


def test_hybrid_ranks_the_matching_skill_first() -> None:
    index = HybridIndex(SKILLS, RetrievalConfig(gate=0.0, cap=3))
    result = index.shortlist("please merge these pdf files")
    assert result.ids[0] == "a/pdf"
    assert list(result.scores) == sorted(result.scores, reverse=True)


def test_gate_and_cap_limit_the_shortlist() -> None:
    wide = HybridIndex(SKILLS, RetrievalConfig(gate=0.0, cap=3)).shortlist("merge pdf")
    assert len(wide.ids) == 3
    gated = HybridIndex(SKILLS, RetrievalConfig(gate=0.9, cap=3)).shortlist("merge pdf")
    assert gated.ids == ("a/pdf",)
    capped = HybridIndex(SKILLS, RetrievalConfig(gate=0.0, cap=2)).shortlist("merge pdf")
    assert len(capped.ids) == 2


def test_description_source_switches_indexed_text() -> None:
    original = HybridIndex(SKILLS, RetrievalConfig(description_source="original", gate=0.0))
    rewritten = HybridIndex(SKILLS, RetrievalConfig(description_source="rewritten", gate=0.0))
    # Only the rewritten xlsx text contains "spreadsheets"; BM25 weight 1 isolates the effect.
    q = "spreadsheets"
    assert original.bm25.get_scores(tokenize(q))[1] == 0
    assert rewritten.bm25.get_scores(tokenize(q))[1] > 0


def test_no_lexical_match_falls_back_to_embeddings() -> None:
    index = HybridIndex(SKILLS, RetrievalConfig(embedding_weight=0.5, gate=0.0, cap=3))
    result = index.shortlist("zzz qqq")
    assert len(result.ids) == 3
    assert 0.0 <= result.top_cosine <= 1.0
