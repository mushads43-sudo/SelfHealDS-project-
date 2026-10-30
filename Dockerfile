# =============================================================================
# SelfHealDS - production image
#   Stage 1 (builder): resolve pinned dependencies into an isolated venv
#   Stage 2 (runtime): copy venv + application only, run as non-root
# Build:  docker build -t selfhealds:local --build-arg APP_VERSION=1.0.0 \
#                      --build-arg GIT_SHA=$(git rev-parse --short HEAD) .
# Run:    docker run -p 8501:8501 -p 8000:8000 selfhealds:local
# Local validation with a different base: --build-arg BASE_IMAGE=<image>
# =============================================================================
ARG BASE_IMAGE=python:3.12-slim

# ---------- Stage 1: dependency builder ----------
FROM ${BASE_IMAGE} AS builder
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
# Copy ONLY the lock file first so this layer is cached until deps change.
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

# ---------- Stage 2: runtime ----------
FROM ${BASE_IMAGE} AS runtime

ARG APP_VERSION=dev
ARG GIT_SHA=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.title="SelfHealDS" \
      org.opencontainers.image.description="Explainable, evidence-driven self-healing data-quality framework (Streamlit)" \
      org.opencontainers.image.version="${APP_VERSION}" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.source="https://github.com/mushads43-sudo/selfhealds"

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/app \
    APP_VERSION=${APP_VERSION} \
    GIT_SHA=${GIT_SHA} \
    SELFHEALDS_METRICS_ENABLED=true \
    SELFHEALDS_METRICS_PORT=8000 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Unprivileged user with a fixed UID (matches runAsUser in k8s/deployment.yaml)
RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --home-dir /home/app --create-home app

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
# Application code + bundled datasets (.dockerignore removes paper/, tests/, infra...)
COPY --chown=app:app . .

USER 10001
EXPOSE 8501 8000

# Streamlit's built-in health endpoint; python is used because slim has no curl.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health',timeout=4).read()==b'ok' else 1)"

# monitoring.launch starts /metrics on :8000, then runs `streamlit run streamlit_app.py`
ENTRYPOINT ["python", "-m", "monitoring.launch"]
CMD ["--server.port=8501", "--server.address=0.0.0.0"]
