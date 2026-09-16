"""Command-line entrypoint. The only module allowed to import every layer.

Each phase registers its own sub-app here (`data`, `tokenizer`, `train`, `translate`, `eval`).
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint

from amnmt import __version__
from amnmt.core.config import load_config

app = typer.Typer(no_args_is_help=True, add_completion=False)
config_app = typer.Typer(help="Inspect and validate experiment configs.")
app.add_typer(config_app, name="config")


@app.command()
def version() -> None:
    rprint(__version__)


@config_app.command("validate")
def config_validate(path: Path) -> None:
    """Load a YAML config and fail on unknown or malformed keys."""
    cfg = load_config(path)
    rprint(cfg.model_dump(mode="json"))


if __name__ == "__main__":
    app()
