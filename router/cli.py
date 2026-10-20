"""Command-line entry point: `router ...`."""

import typer

from router.config import REPO_ROOT, SPEND_LEDGER, load_settings
from router.corpus import collect as collect_mod
from router.corpus import leaderboard, pipeline
from router.llm import LLMClient, ResponseCache
from router.spend import ConfirmationRequired, SpendLimitExceeded, SpendTracker

app = typer.Typer(help="Route prompts to the skills they need.", no_args_is_help=True)
corpus_app = typer.Typer(help="Build the skill corpus.", no_args_is_help=True)
app.add_typer(corpus_app, name="corpus")


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
    yes: bool = typer.Option(
        False, "--yes", help="Approve a run estimated above the confirm threshold"
    ),
) -> None:
    """Apply every curation rule and write corpus/skills.jsonl (makes paid LLM calls)."""
    settings = load_settings()
    tracker = SpendTracker(SPEND_LEDGER, settings.spend_limit_usd)
    client = LLMClient(
        settings.openrouter_api_key, tracker, cache=ResponseCache(pipeline.LLM_CACHE)
    )
    try:
        run_path = pipeline.run_curation(client, tracker, settings.spend_confirm_usd, yes)
    except (SpendLimitExceeded, ConfirmationRequired) as e:
        typer.echo(f"Stopped: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"Curation run saved to {run_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    app()
