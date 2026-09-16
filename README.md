# English ↔ Amharic Transformer NMT (v2)

A from-scratch PyTorch Transformer for bidirectional English–Amharic translation, rebuilt in
verified phases. The plan, layer rules, and milestone checks live in [PLAN.md](PLAN.md).

v1 is preserved at git tag `v1-legacy`.

## Setup

```bash
uv sync --all-extras --group dev
uv run amnmt --help
make check      # ruff + import-linter + mypy + pytest
```

## Status

See the phase checkboxes in [PLAN.md](PLAN.md) and recorded results in `docs/milestones/`.
