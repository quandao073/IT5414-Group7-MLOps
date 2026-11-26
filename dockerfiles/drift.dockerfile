# Drift detection service (contract C14). Build: `uv run invoke docker-build`
# Run:   docker run --rm -p 8001:8001 -v ./data/processed:/app/data/processed:ro \
#          -e DATABASE_URL=postgresql+psycopg://user:password@host:5432/hotel drift:latest
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project

COPY src src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-editable


FROM python:3.12-slim-bookworm

RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    REFERENCE_DIR=/app/data/processed

USER app
EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8001/health', timeout=3)"]

CMD ["uvicorn", "hotel_booking.drift_service:app", "--host", "0.0.0.0", "--port", "8001"]
