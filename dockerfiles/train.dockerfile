# Training job. Build: `uv run invoke docker-build`
# Run:   docker run --rm -v ./data:/app/data -v ./models:/app/models train:latest
# The training entry point is `src/hotel_booking/train.py` (owned by the training module).
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project

COPY src src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-editable


FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY configs configs

ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "hotel_booking.train"]
