.PHONY: help install dev lint format test clean docker-build docker-up run-api run-app

PYTHON ?= python
PIP ?= pip
UVICORN ?= uvicorn
STREAMLIT ?= streamlit

help:
	@echo "Available commands:"
	@echo "  make install        Install production dependencies"
	@echo "  make dev            Install development dependencies & editable package"
	@echo "  make lint           Run linting checks (ruff, black --check)"
	@echo "  make format         Format code using black and ruff"
	@echo "  make test           Run test suite with pytest"
	@echo "  make clean          Clean build artifacts and caches"
	@echo "  make docker-build   Build Docker image"
	@echo "  make docker-up      Run API and UI services via docker-compose"
	@echo "  make run-api        Start FastAPI translation backend"
	@echo "  make run-app        Start Streamlit web application"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -e .

dev:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

lint:
	ruff check src/ tests/ configs/
	black --check src/ tests/

format:
	ruff check --fix src/ tests/ configs/
	black src/ tests/

test:
	pytest tests/ -v --tb=short

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache/ .ruff_cache/ htmlcov/ .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} +

docker-build:
	docker build -t english-amharic-nmt:latest .

docker-up:
	docker compose up --build

run-api:
	$(UVICORN) src.nmt_engine.serving.api:app --host 0.0.0.0 --port 8000 --reload

run-app:
	$(STREAMLIT) run app.py --server.port 8501
