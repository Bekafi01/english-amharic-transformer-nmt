"""Command-line entrypoint. The only module allowed to import every layer.

Each phase registers its own sub-app here (`data`, `tokenizer`, `train`, `translate`, `eval`).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer
from rich import print as rprint

from amnmt import __version__
from amnmt.core.config import Config, load_config

app = typer.Typer(no_args_is_help=True, add_completion=False)
config_app = typer.Typer(help="Inspect and validate experiment configs.")
data_app = typer.Typer(help="Build the parallel corpus (Phase 1).")
tokenizer_app = typer.Typer(help="Train the joint BPE tokenizer (Phase 2).")
app.add_typer(config_app, name="config")
app.add_typer(data_app, name="data")
app.add_typer(tokenizer_app, name="tokenizer")

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
    rebuild: Annotated[
        list[str] | None,
        typer.Option(
            "--rebuild",
            help="Source name to re-stream even if its shard exists; 'all' for every source.",
        ),
    ] = None,
) -> None:
    """Stream, normalize, filter, dedup and split all configured sources."""
    from amnmt.data.pipeline import build

    cfg = _load(config, root, raw_dir)
    card = build(cfg, rebuild=set(rebuild or []))
    rprint(json.dumps({k: card[k] for k in ("sources", "dedup", "splits")}, indent=2))


@tokenizer_app.command("train")
def tokenizer_train(config: ConfigOpt, root: RootOpt = None) -> None:
    """Sample train.parquet and train the joint BPE tokenizer into <artifacts>/tokenizer/."""
    from amnmt.tokenization.train import train

    stats = train(_load(config, root, None))
    rprint(json.dumps(stats, indent=2))


@app.command()
def train(
    config: ConfigOpt,
    root: RootOpt = None,
    run: Annotated[str | None, typer.Option(help="Run name (default: project name).")] = None,
    resume: Annotated[bool, typer.Option(help="Continue from <run>/last.pt if present.")] = False,
    max_steps: Annotated[int | None, typer.Option(help="Override training.max_steps.")] = None,
    time_limit: Annotated[
        int | None, typer.Option(help="Override training.time_limit_minutes.")
    ] = None,
) -> None:
    """Train the Transformer (Phase 4). Checkpoints go to <artifacts>/runs/<run>/."""
    from amnmt.training.trainer import Trainer

    cfg = _load(config, root, None)
    if cfg.training is None:
        raise typer.BadParameter("config has no `training` section")
    overrides: dict[str, int] = {}
    if max_steps is not None:
        overrides["max_steps"] = max_steps
    if time_limit is not None:
        overrides["time_limit_minutes"] = time_limit
    if overrides:
        cfg = cfg.model_copy(update={"training": cfg.training.model_copy(update=overrides)})
    run_dir = cfg.paths.resolve("artifacts") / "runs" / (run or cfg.project.name)
    state = Trainer(cfg, run_dir).train(resume=resume)
    rprint(json.dumps(asdict(state), indent=2))


if __name__ == "__main__":
    app()
