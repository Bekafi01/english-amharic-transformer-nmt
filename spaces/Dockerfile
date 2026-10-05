# Hugging Face Space (sdk: docker). Streamlit on port 7860, CPU-only torch.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/user/.cache/huggingface
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /home/user/app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py .
RUN chown -R user /home/user
USER user
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0", "--server.headless", "true"]
