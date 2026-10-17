# syntax=docker/dockerfile:1
ARG PYTHON_BASE=python:3.12.13-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2
FROM ${PYTHON_BASE} AS runtime
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 10001 heartwise
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/app HOME=/home/heartwise

FROM runtime AS api
COPY requirements-api.lock .
# CPU-only serving does not need XGBoost's optional CUDA NCCL runtime.
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-deps -r requirements-api.lock
COPY src/ src/
COPY api/ api/
ARG BUNDLE_ID
COPY artifacts/${BUNDLE_ID}/ /app/artifacts/${BUNDLE_ID}/
ENV HEARTWISE_ARTIFACT=${BUNDLE_ID}
USER heartwise
RUN python -m src.artifacts validate
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/ready')" || exit 1
CMD ["uvicorn","api.main:app","--host","0.0.0.0","--port","8000","--no-access-log"]

FROM runtime AS streamlit
COPY requirements-ui.lock .
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-deps -r requirements-ui.lock
COPY src/ src/
COPY app/ app/
COPY .streamlit/ .streamlit/
ENV HEARTWISE_MODE=api HEARTWISE_API_URL=http://api:8000
USER heartwise
EXPOSE 8501
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1
CMD ["streamlit","run","app/streamlit_app.py","--server.address=0.0.0.0","--server.port=8501"]
