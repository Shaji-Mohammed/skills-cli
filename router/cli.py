"""Command-line entry point: `router ...`."""

import typer

from router.config import REPO_ROOT, SPEND_LEDGER, load_settings
from router.corpus import collect as collect_mod
from router.corpus import leaderboard, pipeline
from router.corpus import report as corpus_report
from router.evalset import equivalence
from router.llm import LLMClient, ResponseCache
from router.runs import DirtyTree
from router.spend import SpendTracker

app = typer.Typer(help="Route prompts to the skills they need.", no_args_is_help=True)
corpus_app = typer.Typer(help="Build the skill corpus.", no_args_is_help=True)
app.add_typer(corpus_app, name="corpus")
evals_app = typer.Typer(help="Build the eval set.", no_args_is_help=True)
app.add_typer(evals_app, name="evals")


@app.callback()
def main() -> None:
    """Skill router."""


@app.command()
def spend() -> None:
    """Show total API spend against the hard limit."""
    settings = load_settings()
    tracker = SpendTracker(SPEND_LEDGER, settings.spend_limit_usd)
    total = tracker.total()
    typer.echo(f"Spent ${total:.4f} of ${settings.spend_limit_usd:.2f} limit")


@corpus_app.command("fetch-leaderboard")
def fetch_leaderboard(pages: int = typer.Option(5, help="200 skills per page")) -> None:
    """Snapshot the skills.sh all-time leaderboard into corpus/sources/."""
    snapshot = leaderboard.fetch_leaderboard(pages)
    leaderboard.save_snapshot(snapshot, pipeline.SNAPSHOT)
    typer.echo(
        f"Saved {len(snapshot['skills'])} skills to {pipeline.SNAPSHOT.relative_to(REPO_ROOT)}"
    )


@corpus_app.command()
def collect() -> None:
    """Clone source repos and normalize every leaderboard skill into corpus/raw/."""
    skills, failures = collect_mod.collect(leaderboard.load_snapshot(pipeline.SNAPSHOT))
    pipeline.write_jsonl(pipeline.RAW, skills)
    pipeline.write_jsonl(pipeline.RAW_FAILURES, failures)
    typer.echo(f"Collected {len(skills)} skills, {len(failures)} failures")


@corpus_app.command()
def curate(
    allow_dirty: bool = typer.Option(
        False, "--allow-dirty", help="Run from uncommitted code (the run can't be quoted)"
    ),
) -> None:
    """Apply every curation rule and write corpus/skills.jsonl (uses the local LLM)."""
    settings = load_settings()
    client = LLMClient(
        settings.ollama_model, settings.ollama_url, cache=ResponseCache(pipeline.LLM_CACHE)
    )
    try:
        run_path = pipeline.run_curation(client, allow_dirty=allow_dirty)
    except DirtyTree as e:
        typer.echo(f"Stopped: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"Curation run saved to {run_path.relative_to(REPO_ROOT)}")


@corpus_app.command()
def report() -> None:
    """Write corpus/CURATION.md from the latest curation run."""
    run_dir = corpus_report.latest_run()
    (pipeline.CORPUS_DIR / "CURATION.md").write_text(corpus_report.render(run_dir))
    typer.echo(f"Wrote corpus/CURATION.md from {run_dir.relative_to(REPO_ROOT)}")


@evals_app.command("equivalence-candidates")
def equivalence_candidates(
    allow_dirty: bool = typer.Option(
        False, "--allow-dirty", help="Run from uncommitted code (the run can't be quoted)"
    ),
) -> None:
    """Build equivalence candidates and write the review CSV (uses the local LLM)."""
    settings = load_settings()
    client = LLMClient(
        settings.ollama_model, settings.ollama_url, cache=ResponseCache(equivalence.LLM_CACHE)
    )
    try:
        run_path = equivalence.generate(client, allow_dirty=allow_dirty)
    except DirtyTree as e:
        typer.echo(f"Stopped: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(
        f"Review {equivalence.REVIEW_CSV.relative_to(REPO_ROOT)} "
        f"(run {run_path.relative_to(REPO_ROOT)})"
    )


@evals_app.command("freeze-equivalence")
def freeze_equivalence(
    allow_dirty: bool = typer.Option(
        False, "--allow-dirty", help="Run from uncommitted code (the run can't be quoted)"
    ),
) -> None:
    """Merge accepted pairs from the reviewed CSV into evals/equivalence.json."""
    try:
        data, run_path = equivalence.freeze(allow_dirty=allow_dirty)
    except DirtyTree as e:
        typer.echo(f"Stopped: {e}", err=True)
        raise typer.Exit(1) from e
    sizes = [len(s["members"]) for s in data["sets"]]
    typer.echo(
        f"Froze {len(sizes)} sets covering {sum(sizes)} skills "
        f"(run {run_path.relative_to(REPO_ROOT)})"
    )


if __name__ == "__main__":
    app()
