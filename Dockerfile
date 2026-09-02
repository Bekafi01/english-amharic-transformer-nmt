# Base image with Python 3.10 slim
FROM python:3.10-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definition
COPY pyproject.toml .

# Install dependencies and package in editable mode
RUN pip install --upgrade pip && \
    pip install -e .

# Copy project files
COPY configs/ configs/
COPY src/ src/
COPY artifacts/ artifacts/
COPY scripts/ scripts/
COPY app.py .

EXPOSE 8000 8501

# Default entrypoint runs FastAPI server
CMD ["uvicorn", "src.nmt_engine.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
