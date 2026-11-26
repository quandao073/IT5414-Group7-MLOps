"""Prediction API (contracts C1, C7, C9 and C10 in `docs/source/contracts.md`).

Run locally with `uv run invoke serve-api` and open http://localhost:8000/docs.
"""

import logging
import os
import secrets
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request

from hotel_booking.config import GROUP_SIZE_COLUMN, MODEL_INPUT_COLUMNS
from hotel_booking.data import add_group_size, validate_inputs
from hotel_booking.loader import ModelLoadError, ModelSource, ServedModel, source_from_env
from hotel_booking.monitoring import ApiMetrics, instrument
from hotel_booking.prediction_log import PredictionLog, PredictionRecord, connect
from hotel_booking.schemas import (
    LOG_ONLY_FIELDS,
    REPLAY_MONTH_PATTERN,
    BatchRequest,
    BatchResponse,
    Booking,
    HealthResponse,
    InputWarning,
    Prediction,
    PredictResponse,
    ReloadResponse,
)

ReplayMonth = Annotated[
    str | None, Header(pattern=REPLAY_MONTH_PATTERN, description="Replayed month `YYYY-MM`, only for the log.")
]

NULLABLE_FLOATS = ["children", "agent", "company"]
logger = logging.getLogger(__name__)


def bookings_to_frame(bookings: list[Booking]) -> pd.DataFrame:
    """Turn request bookings into the raw frame the model pipeline expects.

    A missing `rooms_in_booking` is counted from identical rows in the same request, as in `data.add_group_size`;
    a single booking therefore defaults to 1.

    Args:
        bookings: Validated bookings.

    Returns:
        A frame with the columns in `MODEL_INPUT_COLUMNS`.
    """
    frame = pd.DataFrame([booking.model_dump(exclude=LOG_ONLY_FIELDS) for booking in bookings])
    frame[NULLABLE_FLOATS] = frame[NULLABLE_FLOATS].astype("float64")
    missing = frame[GROUP_SIZE_COLUMN].isna()
    if missing.any():
        frame[GROUP_SIZE_COLUMN] = frame[GROUP_SIZE_COLUMN].where(~missing, add_group_size(frame))
    frame[GROUP_SIZE_COLUMN] = frame[GROUP_SIZE_COLUMN].astype("int64")
    return frame[MODEL_INPUT_COLUMNS]


def input_warnings(frame: pd.DataFrame, schema: dict[str, Any]) -> list[InputWarning]:
    """Values the model accepts but did not see in train.

    Args:
        frame: Bookings to check.
        schema: The `input_schema` of the model card.

    Returns:
        One warning per column and issue.
    """
    issues = validate_inputs(frame, schema)
    return [InputWarning.model_validate(record) for record in issues.to_dict(orient="records")]


def create_app(
    model_path: Path | None = None,
    card_path: Path | None = None,
    source: ModelSource | None = None,
    reload_token: str | None = None,
    database_url: str | None = None,
) -> FastAPI:
    """Build the API.

    Args:
        model_path: Model file of the default file source; see `loader.source_from_env`.
        card_path: Model card of the default file source.
        source: Model source; overrides `model_path` and `card_path`.
        reload_token: Secret required in the `X-Reload-Token` header of `POST /reload`; defaults to the
            `RELOAD_TOKEN` environment variable. An empty value, or neither being set, leaves `/reload` open.
        database_url: Database for the prediction log; defaults to `DATABASE_URL`. An empty value, or neither
            being set, disables the log.

    Returns:
        The FastAPI application.
    """
    reload_lock = threading.Lock()

    def open_log(app: FastAPI) -> None:
        app.state.prediction_log = None
        url = (database_url if database_url is not None else os.getenv("DATABASE_URL")) or None
        if url is None:
            return
        try:
            app.state.prediction_log = PredictionLog(connect(url), on_error=lambda _: metrics.log_errors.inc())
            logger.info("Writing predictions to %s", url.split("@")[-1])
        except Exception as error:
            metrics.log_errors.inc()
            logger.warning("Prediction log disabled, cannot open the database: %s", error)

    def log_request(
        request: Request,
        background: BackgroundTasks,
        served: ServedModel,
        frame: pd.DataFrame,
        probabilities: list[float],
        bookings: list[Booking],
        replay_month: str | None,
        warnings: list[InputWarning],
        started: float,
    ) -> None:
        prediction_log: PredictionLog | None = request.app.state.prediction_log
        if prediction_log is None:
            return
        record = PredictionRecord(
            endpoint=request.url.path,
            model_version=served.version,
            backend=served.backend,
            threshold=served.threshold,
            latency_ms=(time.perf_counter() - started) * 1000,
            inputs=frame,
            probabilities=probabilities,
            booking_ids=[booking.booking_id for booking in bookings],
            replay_month=replay_month,
            warnings=[warning.model_dump() for warning in warnings],
        )
        background.add_task(prediction_log.write, record)

    def swap_model(app: FastAPI) -> ServedModel:
        served = model_source.load()
        app.state.served = served
        metrics.set_model(served.version, served.backend)
        return served

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.served = None
        metrics.set_model(None)
        try:
            served = swap_model(app)
            logger.info("Serving model %s from %s", served.version, model_source.describe())
        except ModelLoadError as error:
            logger.warning("Starting without a model: %s", error)
        open_log(app)
        yield
        if app.state.prediction_log is not None:
            app.state.prediction_log.engine.dispose()

    model_source = source or source_from_env(model_path, card_path)
    token = (reload_token if reload_token is not None else os.getenv("RELOAD_TOKEN")) or None
    app = FastAPI(
        title="Hotel booking cancellation API",
        description="Predicts the probability that a hotel booking is canceled.",
        version="0.2.0",
        lifespan=lifespan,
    )
    metrics = ApiMetrics()
    app.state.metrics = metrics
    instrument(app, metrics)

    def served_model(request: Request) -> ServedModel:
        served: ServedModel | None = request.app.state.served
        if served is None:
            raise HTTPException(status_code=503, detail="No model loaded. Run `uv run invoke sample-model` first.")
        return served

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        """Report whether a model is loaded and which version."""
        served: ServedModel | None = request.app.state.served
        return HealthResponse(
            status="ok",
            model_loaded=served is not None,
            model_version=served.version if served else None,
            prediction_logging=request.app.state.prediction_log is not None,
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(
        booking: Booking, request: Request, background: BackgroundTasks, x_replay_month: ReplayMonth = None
    ) -> PredictResponse:
        """Predict the cancellation probability of one booking."""
        started = time.perf_counter()
        served = served_model(request)
        frame = bookings_to_frame([booking])
        probability = float(served.predict(frame)[0])
        warnings = input_warnings(frame, served.schema)
        metrics.observe_predictions([probability], served.threshold)
        metrics.observe_warnings(warnings)
        log_request(request, background, served, frame, [probability], [booking], x_replay_month, warnings, started)
        return PredictResponse(
            probability=probability,
            is_canceled=probability >= served.threshold,
            threshold=served.threshold,
            model_version=served.version,
            warnings=warnings,
        )

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(
        batch: BatchRequest, request: Request, background: BackgroundTasks, x_replay_month: ReplayMonth = None
    ) -> BatchResponse:
        """Predict the cancellation probabilities of several bookings, in request order."""
        started = time.perf_counter()
        served = served_model(request)
        frame = bookings_to_frame(batch.bookings)
        probabilities = [float(value) for value in served.predict(frame)]
        warnings = input_warnings(frame, served.schema)
        metrics.observe_predictions(probabilities, served.threshold)
        metrics.observe_warnings(warnings)
        log_request(
            request, background, served, frame, probabilities, batch.bookings, x_replay_month, warnings, started
        )
        return BatchResponse(
            predictions=[
                Prediction(probability=value, is_canceled=value >= served.threshold) for value in probabilities
            ],
            threshold=served.threshold,
            model_version=served.version,
            warnings=warnings,
        )

    @app.post(
        "/reload",
        response_model=ReloadResponse,
        responses={401: {"description": "Wrong or missing X-Reload-Token"}, 503: {"description": "Reload failed"}},
    )
    def reload(request: Request, x_reload_token: str | None = Header(default=None)) -> ReloadResponse:
        """Load the current model from its source without restarting; on failure the previous model stays."""
        if token is not None and not secrets.compare_digest(x_reload_token or "", token):
            raise HTTPException(status_code=401, detail="Wrong or missing X-Reload-Token.")
        with reload_lock:
            previous: ServedModel | None = request.app.state.served
            try:
                served = swap_model(request.app)
            except ModelLoadError as error:
                metrics.reloads.labels(result="failure").inc()
                logger.warning("Reload failed, keeping %s: %s", previous.version if previous else "no model", error)
                raise HTTPException(status_code=503, detail=f"Reload failed, previous model kept. {error}") from error
        metrics.reloads.labels(result="success").inc()
        logger.info("Reloaded model %s (was %s)", served.version, previous.version if previous else None)
        return ReloadResponse(
            previous_version=previous.version if previous else None,
            current_version=served.version,
            changed=previous is None or previous.version != served.version,
            reloaded_at=datetime.now(UTC),
        )

    return app


app = create_app()
