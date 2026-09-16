.PHONY: sync lint fmt arch typecheck test check

sync:
	uv sync --all-extras --group dev

lint:
	uv run ruff check .
	uv run ruff format --check .

fmt:
	uv run ruff format .
	uv run ruff check --fix .

arch:
	uv run lint-imports

typecheck:
	uv run mypy

test:
	uv run pytest -m "not slow"

check: lint arch typecheck test
