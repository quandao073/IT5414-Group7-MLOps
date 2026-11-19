# Streamlit frontend. Run with the rest of the stack: `docker compose up frontend`
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project

COPY src src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-editable


FROM python:3.12-slim-bookworm

# LightGBM and XGBoost need the OpenMP runtime (imported through hotel_booking).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
# frontend.py finds the sample CSV two folders above itself, as in the repo layout.
COPY --chown=app:app src/hotel_booking/frontend.py src/hotel_booking/frontend.py
COPY --chown=app:app docs/examples/stream_sample.csv docs/examples/stream_sample.csv

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    API_URL=http://api:8000

USER app
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3)"]

CMD ["streamlit", "run", "src/hotel_booking/frontend.py", "--server.port=8501", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]
