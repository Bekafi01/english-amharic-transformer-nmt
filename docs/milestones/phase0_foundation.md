# Phase 0 — Foundation

- Commit: `dd3f1c0` (+ import-linter `include_external_packages` fix)
- Date: 2026-09-17

## Check

```
uv sync --group dev
uv run ruff check .        -> All checks passed!
uv run ruff format --check -> 17 files already formatted
uv run lint-imports        -> Contracts: 3 kept, 0 broken
uv run mypy                -> Success: no issues found in 12 source files
uv run pytest              -> 6 passed
```

Result: PASS.
