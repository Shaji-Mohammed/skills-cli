import pytest

from router.score import (
    Item,
    Prediction,
    exact_equivalence,
    exact_strict,
    failure_type,
    precision_recall,
    stage1_recall,
    summarize,
)

EQ = {"tdd": "set:tdd", "test-driven-development": "set:tdd"}
NONE: frozenset[str] = frozenset()


# --- empty gold (no-skill prompts) ---------------------------------------------------------


def test_empty_gold_matches_only_empty_prediction() -> None:
    assert exact_strict([], NONE)
    assert exact_equivalence([], NONE, EQ)
    assert not exact_strict(["pdf"], NONE)
    assert not exact_equivalence(["pdf"], NONE, EQ)


def test_empty_gold_precision_recall() -> None:
    assert precision_recall([], NONE, EQ) == (1.0, 1.0)
    assert precision_recall(["pdf", "xlsx"], NONE, EQ) == (0.0, 1.0)


def test_empty_gold_with_skills_is_overrouting() -> None:
    item = Item("q", "no_skill", "model_drafted", NONE)
    assert failure_type(item, Prediction("q", ()), EQ) is None
    assert failure_type(item, Prediction("q", ("pdf",)), EQ) == "OVERROUTING"


def test_empty_gold_stage1_recall_is_vacuous() -> None:
    assert stage1_recall([], NONE, EQ)


# --- equivalence-aware metric --------------------------------------------------------------


def test_equivalent_swap_counts_only_in_equivalence_metric() -> None:
    gold = frozenset({"tdd"})
    assert not exact_strict(["test-driven-development"], gold)
    assert exact_equivalence(["test-driven-development"], gold, EQ)


def test_two_members_of_one_set_is_overrouting() -> None:
    gold = frozenset({"tdd"})
    pred = ("tdd", "test-driven-development")
    assert not exact_equivalence(pred, gold, EQ)
    item = Item("q", "explicit", "model_drafted", gold)
    assert failure_type(item, Prediction("q", pred), EQ) == "OVERROUTING"


def test_equivalence_with_other_gold_skills() -> None:
    gold = frozenset({"tdd", "pdf"})
    assert exact_equivalence(["pdf", "test-driven-development"], gold, EQ)
    assert not exact_equivalence(["pdf"], gold, EQ)
    assert precision_recall(["pdf"], gold, EQ) == (1.0, 0.5)


def test_skills_outside_any_set_are_compared_directly() -> None:
    assert exact_equivalence(["pdf"], frozenset({"pdf"}), EQ)
    assert not exact_equivalence(["xlsx"], frozenset({"pdf"}), EQ)


# --- failure taxonomy ----------------------------------------------------------------------


def test_failure_types() -> None:
    item = Item("q", "explicit", "model_drafted", frozenset({"pdf", "xlsx"}))
    assert failure_type(item, Prediction("q", (), abstained=True), EQ) == "ABSTAIN"
    assert (
        failure_type(item, Prediction("q", ("pdf",), shortlist=("pdf", "docx")), EQ)
        == "MISS_STAGE1"
    )
    assert (
        failure_type(item, Prediction("q", ("pdf",), shortlist=("pdf", "xlsx")), EQ)
        == "WRONG_SCORER"
    )
    assert failure_type(item, Prediction("q", ("pdf", "xlsx", "docx")), EQ) == "OVERROUTING"


# --- summary -------------------------------------------------------------------------------


def test_summarize_groups_and_intervals() -> None:
    items = [
        Item("a", "explicit", "model_drafted", frozenset({"pdf"})),
        Item("b", "explicit", "model_drafted", frozenset({"pdf", "xlsx"})),
        Item("c", "no_skill", "model_drafted", NONE),
        Item("d", "implied", "hand_written", frozenset({"tdd"})),
    ]
    preds = [
        Prediction("a", ("pdf",), path="fast", latency_ms=10),
        Prediction("b", ("pdf",), path="scorer", latency_ms=30),
        Prediction("c", (), path="scorer", latency_ms=20),
        Prediction("d", ("test-driven-development",), path="fallback", latency_ms=900),
    ]
    out = summarize(items, preds, EQ)
    g = out["groups"]
    assert g["overall"]["exact_strict"]["k"] == 2  # a, c
    assert g["overall"]["exact_equivalence"]["k"] == 3  # a, c, d
    assert g["bucket:explicit"]["exact_equivalence"]["n"] == 2
    assert g["multi_skill"]["exact_equivalence"]["k"] == 0
    assert g["source:hand_written"]["exact_equivalence"]["rate"] == 1.0
    assert g["overall"]["stage1_recall"] is None  # no shortlists reported
    lo, hi = g["overall"]["exact_equivalence"]["wilson95"]
    assert lo < 0.75 < hi
    assert out["failures"]["WRONG_SCORER"] == 1  # b missed xlsx
    assert out["paths"]["fallback"]["share"]["k"] == 1


def test_summarize_requires_every_prediction() -> None:
    with pytest.raises(ValueError):
        summarize([Item("a", "explicit", "model_drafted", NONE)], [], EQ)
