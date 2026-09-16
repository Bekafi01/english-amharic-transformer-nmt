# ==============================================================================
# Multi-Stage Production Dockerfile for English-Amharic NMT REST Service
# ==============================================================================
FROM python:3.12-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install runtime utilities & curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install project dependencies
COPY pyproject.toml README.md ./
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install .

# Copy source tree and configuration files
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Create non-root application user for production security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/checkpoints /app/artifacts /app/data && \
    chown -R appuser:appuser /app

USER appuser

# Expose FastAPI REST service port
EXPOSE 8000

# Container healthcheck querying the /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch Uvicorn ASGI server hosting the NMT service
CMD ["python", "scripts/main.py", "serve", "--host", "0.0.0.0", "--port", "8000", "--device", "cpu"]
