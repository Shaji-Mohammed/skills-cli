from router.evaluate import BaselineConfig, answer, stage1_metrics
from router.retrieval import RetrievalConfig, Shortlist
from router.score import Item

CFG = BaselineConfig(RetrievalConfig(), answer_k=2, answer_gate=0.9, abstain_cosine=0.5)


def sl(ids: tuple[str, ...], scores: tuple[float, ...], top_cos: float = 0.8) -> Shortlist:
    return Shortlist(ids=ids, scores=scores, top_cosine=top_cos, latency_ms=1.0)


def test_answer_takes_top_k_within_gate() -> None:
    p = answer(sl(("a", "b", "c"), (1.0, 0.95, 0.5)), "q", CFG)
    assert p.skills == ("a", "b")
    assert not p.abstained
    p = answer(sl(("a", "b"), (1.0, 0.5)), "q", CFG)
    assert p.skills == ("a",)


def test_answer_abstains_on_low_absolute_similarity() -> None:
    p = answer(sl(("a",), (1.0,), top_cos=0.3), "q", CFG)
    assert p.skills == () and p.abstained and p.path == "abstain"


def test_stage1_metrics_ignore_no_skill_items_and_use_caps() -> None:
    items = [
        Item("1", "explicit", "model_drafted", frozenset({"x"})),
        Item("2", "implied", "hand_written", frozenset({"y"})),
        Item("3", "no_skill", "model_drafted", frozenset()),
    ]
    shortlists = [
        sl((*(f"s{i}" for i in range(15)), "x"), tuple([1.0] * 16)),  # x at rank 16
        sl(("y",), (1.0,)),
        sl(("z",), (1.0,)),
    ]
    m = stage1_metrics(items, shortlists, cap=20, eq={})
    assert m["recall@10"]["k"] == 1 and m["recall@10"]["n"] == 2
    assert m["recall@20"]["k"] == 2
    assert m["implied_recall@10"]["k"] == 1
