"""Drift detection service: compares the inputs logged by the API with the training data (contract C14).

The API writes every scored booking to `prediction_logs` (contract C9). This service reads one window of that table,
by replay month or by `created_at` range, runs `drift.column_drift` against `train.csv` and exposes the result as
JSON, as an Evidently HTML report and as Prometheus gauges on `/metrics`.

Run locally with `uv run invoke serve-drift` and open http://localhost:8001/docs.
"""

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, generate_latest
from pydantic import BaseModel
from sqlalchemy import Engine, create_engine, text

from hotel_booking.config import MODEL_INPUT_COLUMNS, RAW_CATEGORICAL_INPUTS
from hotel_booking.data import read_split
from hotel_booking.drift import DATASET_DRIFT_SHARE, column_drift

DEFAULT_MIN_ROWS = 100


class ColumnDrift(BaseModel):
    """Drift result of one compared column."""

    column: str
    method: str
    threshold: float
    score: float
    drifted: bool


class DriftResponse(BaseModel):
    """Drift of one window of logged inputs against the training data."""

    window: str
    rows: int
    reference_rows: int
    columns: int
    drifted_columns: int
    drift_share: float
    dataset_drift: bool
    details: list[ColumnDrift]


class DriftHealth(BaseModel):
    """Service status."""

    status: str
    reference_rows: int
    database: bool


class DriftMetrics:
    """Prometheus gauges for the latest drift result of each window, in a registry owned by one app."""

    def __init__(self) -> None:
        """Create the registry and the metrics."""
        self.registry = CollectorRegistry()
        labels = ["window"]
        self.share = Gauge("drift_share", "Share of compared columns that drifted", labels, registry=self.registry)
        self.drifted = Gauge("drift_drifted_columns", "Number of drifted columns", labels, registry=self.registry)
        self.rows = Gauge("drift_rows", "Logged rows in the window", labels, registry=self.registry)
        self.dataset = Gauge(
            "drift_dataset_drift",
            "1 when the drift share reaches the dataset threshold",
            labels,
            registry=self.registry,
        )
        self.checked = Gauge(
            "drift_last_check_timestamp_seconds", "Time of the last check", labels, registry=self.registry
        )
        self.checks = Counter("drift_checks", "Drift checks computed", registry=self.registry)

    def record(self, result: DriftResponse, label: str) -> None:
        """Store a drift result.

        Args:
            result: The computed drift of one window.
            label: Value of the `window` label: the replay month, or `custom` for a `created_at` range so that
                arbitrary ranges do not create a new time series each.
        """
        self.share.labels(label).set(result.drift_share)
        self.drifted.labels(label).set(result.drifted_columns)
        self.rows.labels(label).set(result.rows)
        self.dataset.labels(label).set(int(result.dataset_drift))
        self.checked.labels(label).set_to_current_time()
        self.checks.inc()


def _parse_input(value: Any) -> dict[str, Any]:
    """Read one logged input, stored as JSONB (a dict) in PostgreSQL or as text elsewhere.

    Args:
        value: The `input` column of one row.

    Returns:
        The booking as a dict.
    """
    return value if isinstance(value, dict) else json.loads(value)


def load_logged_inputs(
    engine: Engine, month: str | None = None, start: datetime | None = None, end: datetime | None = None
) -> pd.DataFrame:
    """Read the model inputs logged in one window: a replay month, or `start <= created_at < end`.

    Args:
        engine: Connection to the database holding `prediction_logs`.
        month: Replay month `YYYY-MM`.
        start: Start of the `created_at` window, inclusive.
        end: End of the `created_at` window, exclusive.

    Returns:
        One row per logged booking with the columns in `MODEL_INPUT_COLUMNS`.
    """
    params: dict[str, Any]
    if month is not None:
        query, params = "SELECT input FROM prediction_logs WHERE replay_month = :month", {"month": month}
    else:
        query = "SELECT input FROM prediction_logs WHERE created_at >= :start AND created_at < :end"
        params = {"start": start, "end": end}
    with engine.connect() as connection:
        rows = connection.execute(text(f"{query} ORDER BY id"), params).scalars().all()
    frame = pd.DataFrame([_parse_input(row) for row in rows]).reindex(columns=MODEL_INPUT_COLUMNS)
    frame[RAW_CATEGORICAL_INPUTS] = frame[RAW_CATEGORICAL_INPUTS].astype("str")
    return frame


def latest_month(engine: Engine) -> str | None:
    """The most recent replay month in the log.

    Args:
        engine: Connection to the database holding `prediction_logs`.

    Returns:
        The month `YYYY-MM`, or `None` when nothing has been replayed.
    """
    with engine.connect() as connection:
        return connection.execute(text("SELECT max(replay_month) FROM prediction_logs")).scalar()


def compute_drift(reference: pd.DataFrame, current: pd.DataFrame, window: str) -> tuple[DriftResponse, Any]:
    """Compare a window of logged inputs with the reference data.

    Args:
        reference: The training split.
        current: Logged inputs of the window.
        window: Name of the window, returned in the result.

    Returns:
        The drift result and the Evidently snapshot.
    """
    snapshot, table = column_drift(reference, current)
    share = float(table["drifted"].mean())
    result = DriftResponse(
        window=window,
        rows=len(current),
        reference_rows=len(reference),
        columns=len(table),
        drifted_columns=int(table["drifted"].sum()),
        drift_share=round(share, 4),
        dataset_drift=share >= DATASET_DRIFT_SHARE,
        details=[ColumnDrift.model_validate(record) for record in table.to_dict(orient="records")],
    )
    return result, snapshot


def create_app(
    database_url: str | None = None, reference_dir: Path | None = None, min_rows: int | None = None
) -> FastAPI:
    """Build the drift service.

    Args:
        database_url: SQLAlchemy URL of the log database; defaults to `DATABASE_URL`.
        reference_dir: Folder holding `train.csv`; defaults to `REFERENCE_DIR`, then `data/processed`.
        min_rows: Fewest logged rows needed to compute drift; defaults to `DRIFT_MIN_ROWS`, then 100.

    Returns:
        The FastAPI application.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        url = database_url or os.getenv("DATABASE_URL")
        app.state.engine = create_engine(url, pool_pre_ping=True) if url else None
        app.state.reference = read_split(reference_dir or Path(os.getenv("REFERENCE_DIR", "data/processed")), "train")
        app.state.min_rows = min_rows or int(os.getenv("DRIFT_MIN_ROWS", str(DEFAULT_MIN_ROWS)))
        yield
        if app.state.engine is not None:
            app.state.engine.dispose()

    app = FastAPI(
        title="Hotel booking drift service",
        description="Data drift of the inputs logged by the prediction API against the training data.",
        version="0.1.0",
        lifespan=lifespan,
    )
    metrics = DriftMetrics()

    def engine_of(request: Request) -> Engine:
        engine: Engine | None = request.app.state.engine
        if engine is None:
            raise HTTPException(status_code=503, detail="DATABASE_URL is not set, so there is no prediction log.")
        return engine

    def window_of(month: str | None, start: datetime | None, end: datetime | None) -> str:
        if month is not None and start is None and end is None:
            return month
        if month is None and start is not None and end is not None and start < end:
            return f"{start.isoformat()}/{end.isoformat()}"
        raise HTTPException(status_code=422, detail="Give either `month`, or both `start` and `end` with start < end.")

    def drift_of(
        request: Request, month: str | None, start: datetime | None, end: datetime | None
    ) -> tuple[DriftResponse, Any]:
        window = window_of(month, start, end)
        current = load_logged_inputs(engine_of(request), month, start, end)
        if len(current) < request.app.state.min_rows:
            raise HTTPException(
                status_code=404,
                detail=f"Window {window} has {len(current)} logged rows; at least {request.app.state.min_rows} needed.",
            )
        result, snapshot = compute_drift(request.app.state.reference, current, window)
        metrics.record(result, label=month or "custom")
        return result, snapshot

    @app.get("/health", response_model=DriftHealth)
    def health(request: Request) -> DriftHealth:
        """Report the size of the reference data and whether a log database is configured."""
        return DriftHealth(
            status="ok", reference_rows=len(request.app.state.reference), database=request.app.state.engine is not None
        )

    @app.get("/drift", response_model=DriftResponse)
    def drift(
        request: Request, month: str | None = None, start: datetime | None = None, end: datetime | None = None
    ) -> DriftResponse:
        """Drift of one window: a replay `month` (`YYYY-MM`), or the `created_at` range [`start`, `end`)."""
        return drift_of(request, month, start, end)[0]

    @app.get("/drift/latest", response_model=DriftResponse)
    def drift_latest(request: Request) -> DriftResponse:
        """Drift of the most recent replay month in the log."""
        month = latest_month(engine_of(request))
        if month is None:
            raise HTTPException(status_code=404, detail="No replayed month in the prediction log yet.")
        return drift_of(request, month, None, None)[0]

    @app.get("/drift/report", response_class=HTMLResponse)
    def drift_report(
        request: Request, month: str | None = None, start: datetime | None = None, end: datetime | None = None
    ) -> HTMLResponse:
        """The Evidently HTML report of one window, with the same parameters as `/drift`."""
        return HTMLResponse(drift_of(request, month, start, end)[1].get_html_str(as_iframe=False))

    @app.get("/metrics")
    def prometheus_metrics() -> Response:
        """Prometheus metrics: the latest drift result of each checked window."""
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
