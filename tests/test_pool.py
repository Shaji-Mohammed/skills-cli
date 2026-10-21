import csv
import random
from pathlib import Path
from typing import Any

import pytest

from router.evalset.pool import REVIEW_COLUMNS, reviewed_items, stratified_split


def item(i: int, bucket: str, source: str, multi: bool) -> dict[str, Any]:
    return {
        "id": f"{bucket}-{source}-{i}",
        "bucket": bucket,
        "source": source,
        "multi_skill": multi,
    }


def test_stratified_split_is_exact_proportional_and_seeded() -> None:
    items = (
        [item(i, "explicit", "model_drafted", False) for i in range(200)]
        + [item(i, "implied", "hand_written", False) for i in range(20)]
        + [item(i, "no_skill", "model_drafted", False) for i in range(80)]
    )
    dev, test = stratified_split(items, 60, random.Random(1))
    assert len(test) == 60 and len(dev) == 240
    by_bucket = {
        b: sum(it["bucket"] == b for it in test) for b in ("explicit", "implied", "no_skill")
    }
    assert by_bucket == {"explicit": 40, "implied": 4, "no_skill": 16}
    _, again = stratified_split(items, 60, random.Random(1))
    assert [it["id"] for it in test] == [it["id"] for it in again]
    assert not {it["id"] for it in dev} & {it["id"] for it in test}


def write_sheet(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in REVIEW_COLUMNS})


EQ = {"o/w": "eq:a/s", "a/s": "eq:a/s"}
KNOWN = {"a/s", "o/w", "p/d", "x/l"}


def test_review_decisions_and_missing_skill(tmp_path: Path) -> None:
    sheet = tmp_path / "r.csv"
    write_sheet(
        sheet,
        [
            {
                "id": "1",
                "bucket": "explicit",
                "source": "model_drafted",
                "prompt": "p",
                "proposed_gold": "p/d",
                "decision": "accept",
                "missing_skill": "x/l",
            },
            {
                "id": "2",
                "bucket": "explicit",
                "source": "model_drafted",
                "prompt": "p",
                "proposed_gold": "p/d",
                "decision": "edit",
                "edited_gold": "a/s",
            },
            {
                "id": "3",
                "bucket": "no_skill",
                "source": "model_drafted",
                "prompt": "p",
                "proposed_gold": "",
                "decision": "accept",
            },
            {
                "id": "4",
                "bucket": "explicit",
                "source": "model_drafted",
                "prompt": "p",
                "proposed_gold": "p/d",
                "decision": "drop",
            },
        ],
    )
    items, dropped = reviewed_items(sheet, "full", KNOWN, EQ)
    assert [(it["id"], it["gold"], it["multi_skill"]) for it in items] == [
        ("1", ["p/d", "x/l"], True),
        ("2", ["a/s"], False),
        ("3", [], False),
    ]
    assert [d["id"] for d in dropped] == ["4"]


def test_review_rejects_missing_decision_and_non_canonical(tmp_path: Path) -> None:
    sheet = tmp_path / "r.csv"
    write_sheet(sheet, [{"id": "1", "bucket": "explicit", "proposed_gold": "p/d"}])
    with pytest.raises(ValueError, match="accept/edit/drop"):
        reviewed_items(sheet, "full", KNOWN, EQ)
    write_sheet(
        sheet, [{"id": "1", "bucket": "explicit", "proposed_gold": "o/w", "decision": "accept"}]
    )
    with pytest.raises(ValueError, match="canonical"):
        reviewed_items(sheet, "full", KNOWN, EQ)
