"""Export a trainer checkpoint to a small inference-only bundle.

Trainer checkpoints carry optimizer/scheduler/RNG state (~700 MB for base). The exported file keeps
only what `Translator` needs, optionally in fp16 (~120 MB), plus a copy of tokenizer.json so the
bundle directory is self-contained and uploadable to the Hugging Face Hub as-is.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch

from amnmt import __version__
from amnmt.inference.translator import default_tokenizer_path

EXPORT_KEYS = ("model", "config", "vocab_size", "pad_id")


def export_checkpoint(
    checkpoint: Path, out_dir: Path, tokenizer: Path | None = None, fp16: bool = True
) -> dict[str, Any]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    if fp16:
        state = {k: v.half() if v.is_floating_point() else v for k, v in state.items()}
    # Only model/data.normalize are needed at inference; drop training/data sources etc.
    cfg = ckpt["config"]
    slim_cfg = {
        "project": cfg.get("project"),
        "model": cfg["model"],
        "data": {"normalize": (cfg.get("data") or {}).get("normalize")},
    }
    bundle = {
        "model": state,
        "config": slim_cfg,
        "vocab_size": ckpt["vocab_size"],
        "pad_id": ckpt["pad_id"],
        "export": {
            "amnmt_version": __version__,
            "source": str(checkpoint),
            "step": (ckpt.get("state") or {}).get("step"),
            "best_holdout_nll": (ckpt.get("state") or {}).get("best_holdout_nll"),
            "dtype": "float16" if fp16 else "float32",
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.pt"
    torch.save(bundle, model_path)
    tok_src = Path(tokenizer) if tokenizer else default_tokenizer_path(checkpoint)
    shutil.copyfile(tok_src, out_dir / "tokenizer.json")
    (out_dir / "export.json").write_text(json.dumps(bundle["export"], indent=2), encoding="utf-8")
    return {
        **bundle["export"],
        "model_mb": round(model_path.stat().st_size / 1e6, 1),
        "out_dir": str(out_dir),
    }
