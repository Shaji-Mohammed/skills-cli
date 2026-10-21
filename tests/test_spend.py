from pathlib import Path

import pytest

from router.spend import (
    ConfirmationRequired,
    Price,
    SpendLimitExceeded,
    SpendTracker,
)


def test_empty_ledger_is_zero(tmp_path: Path) -> None:
    assert SpendTracker(tmp_path / "ledger.jsonl", 5.0).total() == 0.0


def test_settled_cost_replaces_estimate(tmp_path: Path) -> None:
    t = SpendTracker(tmp_path / "ledger.jsonl", 5.0)
    call = t.reserve("m", 0.10, "test")
    t.settle(call, 0.02, {})
    assert t.total() == pytest.approx(0.02)


def test_unsettled_reservation_keeps_counting(tmp_path: Path) -> None:
    t = SpendTracker(tmp_path / "ledger.jsonl", 5.0)
    t.reserve("m", 0.10, "test")
    assert t.total() == pytest.approx(0.10)


def test_total_persists_across_instances(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    call = SpendTracker(ledger, 5.0).reserve("m", 1.0, "test")
    SpendTracker(ledger, 5.0).settle(call, 0.75, {})
    assert SpendTracker(ledger, 5.0).total() == pytest.approx(0.75)


def test_reserve_refuses_past_limit(tmp_path: Path) -> None:
    t = SpendTracker(tmp_path / "ledger.jsonl", 1.0)
    t.settle(t.reserve("m", 0.9, "test"), 0.9, {})
    with pytest.raises(SpendLimitExceeded):
        t.reserve("m", 0.2, "test")
    # The refused call must not be recorded.
    assert t.total() == pytest.approx(0.9)


def test_reserve_allows_exactly_hitting_limit(tmp_path: Path) -> None:
    t = SpendTracker(tmp_path / "ledger.jsonl", 1.0)
    t.reserve("m", 1.0, "test")
    assert t.total() == pytest.approx(1.0)


def test_run_budget_requires_confirmation_above_threshold(tmp_path: Path) -> None:
    t = SpendTracker(tmp_path / "ledger.jsonl", 5.0)
    t.check_run_budget(0.40, confirm_usd=0.50, confirmed=False)
    with pytest.raises(ConfirmationRequired):
        t.check_run_budget(0.60, confirm_usd=0.50, confirmed=False)
    t.check_run_budget(0.60, confirm_usd=0.50, confirmed=True)


def test_run_budget_refuses_past_limit_even_if_confirmed(tmp_path: Path) -> None:
    t = SpendTracker(tmp_path / "ledger.jsonl", 1.0)
    with pytest.raises(SpendLimitExceeded):
        t.check_run_budget(1.5, confirm_usd=0.50, confirmed=True)


def test_price_cost() -> None:
    price = Price(input_per_mtok=0.15, output_per_mtok=0.60)
    assert price.cost(1_000_000, 1_000_000) == pytest.approx(0.75)
