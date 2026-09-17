"""Command-line entrypoint. The only module allowed to import every layer.

Each phase registers its own sub-app here (`data`, `tokenizer`, `train`, `translate`, `eval`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich import print as rprint

from amnmt import __version__
from amnmt.core.config import Config, load_config

app = typer.Typer(no_args_is_help=True, add_completion=False)
config_app = typer.Typer(help="Inspect and validate experiment configs.")
data_app = typer.Typer(help="Build the parallel corpus (Phase 1).")
app.add_typer(config_app, name="config")
app.add_typer(data_app, name="data")

ConfigOpt = Annotated[Path, typer.Option("--config", "-c", help="YAML config path.")]
RootOpt = Annotated[
    Path | None, typer.Option("--root", help="Override paths.root (e.g. a Drive folder).")
]
RawDirOpt = Annotated[
    Path | None, typer.Option("--raw-dir", help="Override paths.data_raw (downloads).")
]


def _load(config: Path, root: Path | None, raw_dir: Path | None) -> Config:
    cfg = load_config(config)
    overrides: dict[str, Path] = {}
    if root is not None:
        overrides["root"] = root
    if raw_dir is not None:
        overrides["data_raw"] = raw_dir
    return cfg.with_paths(**overrides) if overrides else cfg


@app.command()
def version() -> None:
    rprint(__version__)


@config_app.command("validate")
def config_validate(path: Path) -> None:
    """Load a YAML config and fail on unknown or malformed keys."""
    cfg = load_config(path)
    rprint(cfg.model_dump(mode="json"))


@data_app.command("build")
def data_build(
    config: ConfigOpt,
    root: RootOpt = None,
    raw_dir: RawDirOpt = None,
    rebuild_shards: Annotated[
        bool, typer.Option(help="Re-stream sources even if their shard already exists.")
    ] = False,
) -> None:
    """Stream, normalize, filter, dedup and split all configured sources."""
    from amnmt.data.pipeline import build

    cfg = _load(config, root, raw_dir)
    card = build(cfg, skip_existing_shards=not rebuild_shards)
    rprint(json.dumps({k: card[k] for k in ("sources", "dedup", "splits")}, indent=2))


if __name__ == "__main__":
    app()
