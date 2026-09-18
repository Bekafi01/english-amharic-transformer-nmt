# CPU inference image. Model files are mounted at runtime (see docker-compose.yml).
#
#   docker build -t amnmt .
#   docker run -p 8000:8000 -v /path/to/artifacts/full:/models -e AMNMT_CHECKPOINT=/models/runs/base_medium/best.pt amnmt

FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

FROM base AS deps
COPY pyproject.toml README.md ./
COPY src ./src
# CPU-only torch keeps the image ~1 GB instead of ~5 GB.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.3" && \
    pip install ".[tokenization,serving]"

FROM base AS runtime
COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin
COPY src ./src
COPY app.py ./
RUN pip install --no-deps -e . && useradd --create-home --uid 1000 amnmt && chown -R amnmt /app
USER amnmt
ENV AMNMT_DEVICE=cpu AMNMT_CHECKPOINT=/models/runs/base_medium/best.pt
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"
CMD ["uvicorn", "amnmt.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
