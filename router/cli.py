"""Command-line entry point: `router ...`."""

import typer

from router import evaluate
from router.config import REPO_ROOT, SPEND_LEDGER, load_settings
from router.corpus import collect as collect_mod
from router.corpus import leaderboard, pipeline, product_locked
from router.corpus import report as corpus_report
from router.evalset import drafts, equivalence, handwritten, pool, slots
from router.llm import LLMClient, ResponseCache
from router.retrieval import RetrievalConfig
from router.runs import DirtyTree
from router.spend import SpendTracker

app = typer.Typer(help="Route prompts to the skills they need.", no_args_is_help=True)
corpus_app = typer.Typer(help="Build the skill corpus.", no_args_is_help=True)
app.add_typer(corpus_app, name="corpus")
evals_app = typer.Typer(help="Build the eval set.", no_args_is_help=True)
app.add_typer(evals_app, name="evals")
eval_app = typer.Typer(help="Evaluate routers on dev.", no_args_is_help=True)
app.add_typer(eval_app, name="eval")


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


@corpus_app.command("product-locked-prescreen")
def product_locked_prescreen(
    allow_dirty: bool = typer.Option(
        False, "--allow-dirty", help="Run from uncommitted code (the run can't be quoted)"
    ),
) -> None:
    """Have qwen flag product-locked skills into a review sheet (uses the local LLM)."""
    settings = load_settings()
    client = LLMClient(
        settings.ollama_model, settings.ollama_url, cache=ResponseCache(product_locked.LLM_CACHE)
    )
    try:
        run_path = product_locked.prescreen(client, allow_dirty=allow_dirty)
    except DirtyTree as e:
        typer.echo(f"Stopped: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(
        f"Review {product_locked.REVIEW_CSV.relative_to(REPO_ROOT)} "
        f"(run {run_path.relative_to(REPO_ROOT)})"
    )


@corpus_app.command("freeze-product-locked")
def freeze_product_locked() -> None:
    """Write the confirmed product-locked list to corpus/product_locked.json."""
    locked = product_locked.freeze()
    typer.echo(f"Froze {len(locked)} product-locked skills")


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


@evals_app.command("handwritten-slots")
def handwritten_slots(
    allow_dirty: bool = typer.Option(
        False, "--allow-dirty", help="Run from uncommitted code (the run can't be quoted)"
    ),
) -> None:
    """Draw the 50 target slots for hand-written implied prompts (uses the local LLM)."""
    settings = load_settings()
    client = LLMClient(
        settings.ollama_model, settings.ollama_url, cache=ResponseCache(equivalence.LLM_CACHE)
    )
    try:
        run_path = slots.build(client, allow_dirty=allow_dirty)
    except DirtyTree as e:
        typer.echo(f"Stopped: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(
        f"Wrote {slots.SHEET.relative_to(REPO_ROOT)} (run {run_path.relative_to(REPO_ROOT)})"
    )


@evals_app.command("redraw-handwritten")
def redraw_handwritten(
    singles: int = typer.Option(8, help="Replacement single-skill slots"),
    pairs: int = typer.Option(6, help="Replacement pair slots"),
    min_cross: int = typer.Option(3, help="Cross-repo pairs to draw first"),
    allow_dirty: bool = typer.Option(
        False, "--allow-dirty", help="Run from uncommitted code (the run can't be quoted)"
    ),
) -> None:
    """Mark swaps from evals/handwritten/swaps.json and append replacement reserve slots."""
    settings = load_settings()
    client = LLMClient(
        settings.ollama_model, settings.ollama_url, cache=ResponseCache(equivalence.LLM_CACHE)
    )
    try:
        run_path = slots.redraw(client, singles, pairs, min_cross, allow_dirty=allow_dirty)
    except DirtyTree as e:
        typer.echo(f"Stopped: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(
        f"Updated {slots.SHEET.relative_to(REPO_ROOT)} (run {run_path.relative_to(REPO_ROOT)})"
    )


@evals_app.command("select-handwritten")
def select_handwritten(
    singles: int = typer.Option(18, help="Single-skill slots to write"),
    pairs: int = typer.Option(7, help="Pair slots to write"),
    allow_dirty: bool = typer.Option(
        False, "--allow-dirty", help="Run from uncommitted code (the run can't be quoted)"
    ),
) -> None:
    """Mark which slots to write (use=yes); the rest become unused."""
    try:
        run_path = slots.select(singles, pairs, allow_dirty=allow_dirty)
    except DirtyTree as e:
        typer.echo(f"Stopped: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(
        f"Updated {slots.SHEET.relative_to(REPO_ROOT)} (run {run_path.relative_to(REPO_ROOT)})"
    )


@evals_app.command()
def draft(
    allow_dirty: bool = typer.Option(
        False, "--allow-dirty", help="Run from uncommitted code (the run can't be quoted)"
    ),
) -> None:
    """Draft 200 explicit, 95 implied and 80 no-skill prompts with qwen (uses the local LLM)."""
    settings = load_settings()
    client = LLMClient(
        settings.ollama_model, settings.ollama_url, cache=ResponseCache(equivalence.LLM_CACHE)
    )
    try:
        run_path = drafts.generate(client, allow_dirty=allow_dirty)
    except DirtyTree as e:
        typer.echo(f"Stopped: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"Wrote {drafts.OUT.relative_to(REPO_ROOT)} (run {run_path.relative_to(REPO_ROOT)})")


@evals_app.command("pool-split")
def pool_split(
    allow_dirty: bool = typer.Option(
        False, "--allow-dirty", help="Run from uncommitted code (the run can't be quoted)"
    ),
) -> None:
    """Pool prompts, drop near-duplicates, split dev/test and write the review sheets."""
    try:
        run_path = pool.build(allow_dirty=allow_dirty)
    except DirtyTree as e:
        typer.echo(f"Stopped: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(
        f"Review {pool.TEST_REVIEW.relative_to(REPO_ROOT)} and "
        f"{pool.DEV_REVIEW.relative_to(REPO_ROOT)} (run {run_path.relative_to(REPO_ROOT)})"
    )


@evals_app.command("apply-review")
def apply_review(
    allow_dirty: bool = typer.Option(
        False, "--allow-dirty", help="Run from uncommitted code (the run can't be quoted)"
    ),
) -> None:
    """Write evals/dev.jsonl and lock evals/test.jsonl from the reviewed sheets."""
    try:
        run_path, sha = pool.apply_review(allow_dirty=allow_dirty)
    except DirtyTree as e:
        typer.echo(f"Stopped: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"Locked evals/test.jsonl sha256={sha} (run {run_path.relative_to(REPO_ROOT)})")


@eval_app.command("stage1-sweep")
def stage1_sweep(
    allow_dirty: bool = typer.Option(
        False, "--allow-dirty", help="Run from uncommitted code (the run can't be quoted)"
    ),
) -> None:
    """Stage 1 recall@10/20/40 on dev for every model x weight x description source."""
    try:
        run_path = evaluate.run_stage1_sweep(allow_dirty=allow_dirty)
    except DirtyTree as e:
        typer.echo(f"Stopped: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"Saved {run_path.relative_to(REPO_ROOT)}")


@eval_app.command("baseline-sweep")
def baseline_sweep(
    model: str = typer.Option("BAAI/bge-small-en-v1.5"),
    source: str = typer.Option("original", help="original | rewritten"),
    weight: float = typer.Option(0.7, help="Embedding weight in the hybrid score"),
    allow_dirty: bool = typer.Option(
        False, "--allow-dirty", help="Run from uncommitted code (the run can't be quoted)"
    ),
) -> None:
    """Grid over the baseline's answer settings on dev, for one retriever config."""
    if source not in ("original", "rewritten"):
        raise typer.BadParameter("source must be original or rewritten")
    cfg = RetrievalConfig(
        embedding_model=model,
        description_source="original" if source == "original" else "rewritten",
        embedding_weight=weight,
    )
    try:
        run_path = evaluate.sweep_baseline(cfg, allow_dirty=allow_dirty)
    except DirtyTree as e:
        typer.echo(f"Stopped: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"Saved {run_path.relative_to(REPO_ROOT)}")


@evals_app.command("check-handwritten")
def check_handwritten() -> None:
    """Run filled hand-written prompts through the D1-D4 filters and label checks."""
    total, results = handwritten.check_sheet(slots.SHEET)
    failing = [r for r in results if r["problems"]]
    for r in failing:
        typer.echo(f"slot {r['slot']}:")
        for p in r["problems"]:
            typer.echo(f"  - {p}")
    typer.echo(f"{len(results)} of {total} rows filled; {len(failing)} with problems")


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
