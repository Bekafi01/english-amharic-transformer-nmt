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


@app.command()
def translate(
    checkpoint: Annotated[Path, typer.Option("--checkpoint", "-m", help="best.pt / last.pt")],
    direction: Annotated[str, typer.Option("--direction", "-d", help="en-am or am-en")],
    text: Annotated[list[str] | None, typer.Argument(help="Sentences to translate.")] = None,
    input_file: Annotated[
        Path | None, typer.Option("--input", "-i", help="File with one sentence per line.")
    ] = None,
    tokenizer: Annotated[
        Path | None, typer.Option(help="tokenizer.json (default: <artifacts>/tokenizer/).")
    ] = None,
    beam: Annotated[int, typer.Option(help="Beam size; 1 = greedy.")] = 4,
    length_penalty: float = 1.0,
) -> None:
    """Translate sentences with a trained checkpoint (Phase 5)."""
    from amnmt.inference.translator import Translator

    sentences = list(text or [])
    if input_file is not None:
        sentences += input_file.read_text(encoding="utf-8").splitlines()
    if not sentences:
        raise typer.BadParameter("give sentences as arguments or via --input")
    if direction not in ("en-am", "am-en"):
        raise typer.BadParameter("direction must be en-am or am-en")
    tr = Translator.from_checkpoint(checkpoint, tokenizer)
    for out in tr.translate(sentences, direction, beam_size=beam, length_penalty=length_penalty):  # type: ignore[arg-type]
        print(out)


@app.command("eval")
def evaluate(
    checkpoint: Annotated[Path, typer.Option("--checkpoint", "-m", help="best.pt / last.pt")],
    config: ConfigOpt,
    root: RootOpt = None,
    split: Annotated[
        str, typer.Option(help="test (FLORES devtest) or valid (FLORES dev)")
    ] = "test",
    beam: int = 4,
    length_penalty: float = 1.0,
    spbleu: Annotated[bool, typer.Option(help="Also compute spBLEU (downloads SPM once).")] = False,
    limit: Annotated[int | None, typer.Option(help="Only the first N sentences (smoke).")] = None,
    tokenizer: Path | None = None,
) -> None:
    """Score a checkpoint on FLORES-200 in both directions (Phase 6). Writes JSON + Markdown
    next to the checkpoint."""
    from amnmt.evaluation.flores import run_and_save

    cfg = _load(config, root, None)
    if split not in ("test", "valid"):
        raise typer.BadParameter("split must be test or valid")
    results = run_and_save(
        checkpoint,
        cfg.paths.resolve("data_processed") / f"{split}.parquet",
        checkpoint.parent,
        tokenizer,
        beam_size=beam,
        length_penalty=length_penalty,
        spbleu=spbleu,
        limit=limit,
    )
    rprint(
        json.dumps({d: r["vs_normalized_ref"] for d, r in results["directions"].items()}, indent=2)
    )


if __name__ == "__main__":
    app()
