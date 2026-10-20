"""Command-line entry point: `router ...`."""

import typer

from router.config import SPEND_LEDGER, load_settings
from router.spend import SpendTracker

app = typer.Typer(help="Route prompts to the skills they need.", no_args_is_help=True)


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


if __name__ == "__main__":
    app()
