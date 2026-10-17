# One image serves the API and the built frontend (plan §4, D5).
#   docker build -t medvisionai .
#   docker run -p 7860:7860 medvisionai   ->  http://localhost:7860

# --- Stage 1: build the React frontend -------------------------------------
FROM node:24-slim AS frontend
WORKDIR /src/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# --- Stage 2: Python runtime (ONNX Runtime + NumPy, no TensorFlow) ---------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN pip install --no-cache-dir uv==0.11.32 \
    && useradd --create-home --uid 1000 app

WORKDIR /app/backend
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY backend/app ./app

# Model files come from a GitHub release and are checksum-verified against
# model_meta.json (D6, D18); the build fails if any file does not match.
ARG MODEL_BASE_URL=https://github.com/OmkarJayvantJadhav/MedVisionAI/releases/download/model-v2.2.0
RUN mkdir -p /app/models && chown app:app /app/models
USER app
RUN python -m app.inference.download --base-url "$MODEL_BASE_URL" --dest /app/models

COPY --from=frontend --chown=app:app /src/frontend/dist /app/static

ENV MEDVISION_MODEL_DIR=/app/models \
    MEDVISION_STATIC_DIR=/app/static \
    PORT=7860
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/v1/health', timeout=4)"

CMD ["sh", "-c", "exec uvicorn app.main:create_app --factory --host 0.0.0.0 --port \"$PORT\" --no-access-log"]
