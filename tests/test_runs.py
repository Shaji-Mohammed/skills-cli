import json
import subprocess
from pathlib import Path

import pytest

from router.runs import file_sha256, git_state, start_run
from router.splits import TestSetLocked, load_split


def test_start_run_writes_manifest(tmp_path: Path) -> None:
    corpus = tmp_path / "skills.jsonl"
    corpus.write_text('{"id": "a"}\n')
    run = start_run(
        "smoke",
        {"cap": 20},
        corpus_path=corpus,
        evalset_path=None,
        results_dir=tmp_path / "results",
        repo=tmp_path,
    )
    run.write_metrics({"recall": 1.0})
    run.write_predictions([{"id": 1}])

    manifest = json.loads((run.path / "run.json").read_text())
    assert manifest["config"] == {"cap": 20}
    assert manifest["corpus_version"] == file_sha256(corpus)
    assert manifest["evalset_version"] is None
    # tmp_path is not a git repo, so the run must be marked unquotable.
    assert manifest["git_commit"] == "unknown"
    assert manifest["git_dirty"] is True
    assert json.loads((run.path / "metrics.json").read_text()) == {"recall": 1.0}
    assert (run.path / "predictions.jsonl").read_text() == '{"id": 1}\n'


def test_git_state_clean_and_dirty(tmp_path: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init")
    (tmp_path / "f").write_text("x")
    git("add", "f")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-m", "init")
    sha, dirty = git_state(tmp_path)
    assert len(sha) == 40
    assert not dirty

    (tmp_path / "f").write_text("y")
    assert git_state(tmp_path)[1]


def test_test_split_is_locked() -> None:
    with pytest.raises(TestSetLocked):
        load_split("test")
