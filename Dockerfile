# Research Paper Q&A: API, browser client, and retrieval stack.
#
# Ollama is deliberately NOT in this image. It is a separate service with
# its own release cadence, its own model store measured in gigabytes, and
# the only component that wants a GPU. Keeping it out means this image
# stays small enough to rebuild quickly, and the two can be scaled or
# placed independently.

FROM python:3.12-slim AS base

# PyMuPDF needs libgomp at runtime. Everything else here is pure wheels.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Model weights land here so a mounted volume can persist them across
    # container restarts instead of re-downloading ~200 MB each time.
    HF_HOME=/models/huggingface \
    OLLAMA_HOST=http://ollama:11434

WORKDIR /app

# Dependencies first, so editing source does not invalidate the layer
# that takes minutes to build.
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY src/ ./src/
COPY eval/splits.json eval/benchmark.json eval/corpus_queries.json ./eval/
COPY scripts/ ./scripts/
COPY app.py ./

# Non-root. The writable paths are volumes, so ownership is set on the
# mount points rather than the whole tree.
RUN useradd --create-home --uid 10001 rag \
 && mkdir -p /data/corpus /data/uploads /models/huggingface \
 && chown -R rag:rag /data /models
USER rag

VOLUME ["/data", "/models"]
EXPOSE 8000

# Readiness, not liveness: the pipeline loads models at startup and the
# endpoint reports ready=false until that finishes, so an orchestrator
# will not route traffic to a container that cannot answer yet.
HEALTHCHECK --interval=15s --timeout=5s --start-period=180s --retries=20 \
  CMD curl -fsS http://localhost:8000/health | grep -q '"ready":true' || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
