"""Prediction API (contracts C1, C7 and C10 in `docs/source/contracts.md`).

Run locally with `uv run invoke serve-api` and open http://localhost:8000/docs.
"""

import logging
import os
import secrets
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request

from hotel_booking.config import GROUP_SIZE_COLUMN, MODEL_INPUT_COLUMNS
from hotel_booking.data import add_group_size, validate_inputs
from hotel_booking.loader import ModelLoadError, ModelSource, ServedModel, source_from_env
from hotel_booking.monitoring import ApiMetrics, instrument
from hotel_booking.schemas import (
    BatchRequest,
    BatchResponse,
    Booking,
    HealthResponse,
    InputWarning,
    Prediction,
    PredictResponse,
    ReloadResponse,
)

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
    frame = pd.DataFrame([booking.model_dump() for booking in bookings])
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
) -> FastAPI:
    """Build the API.

    Args:
        model_path: Model file of the default file source; see `loader.source_from_env`.
        card_path: Model card of the default file source.
        source: Model source; overrides `model_path` and `card_path`.
        reload_token: Secret required in the `X-Reload-Token` header of `POST /reload`; defaults to the
            `RELOAD_TOKEN` environment variable. An empty value, or neither being set, leaves `/reload` open.

    Returns:
        The FastAPI application.
    """
    reload_lock = threading.Lock()

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
        yield

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
            status="ok", model_loaded=served is not None, model_version=served.version if served else None
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(booking: Booking, request: Request) -> PredictResponse:
        """Predict the cancellation probability of one booking."""
        served = served_model(request)
        frame = bookings_to_frame([booking])
        probability = float(served.predict(frame)[0])
        warnings = input_warnings(frame, served.schema)
        metrics.observe_predictions([probability], served.threshold)
        metrics.observe_warnings(warnings)
        return PredictResponse(
            probability=probability,
            is_canceled=probability >= served.threshold,
            threshold=served.threshold,
            model_version=served.version,
            warnings=warnings,
        )

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(batch: BatchRequest, request: Request) -> BatchResponse:
        """Predict the cancellation probabilities of several bookings, in request order."""
        served = served_model(request)
        frame = bookings_to_frame(batch.bookings)
        probabilities = [float(value) for value in served.predict(frame)]
        warnings = input_warnings(frame, served.schema)
        metrics.observe_predictions(probabilities, served.threshold)
        metrics.observe_warnings(warnings)
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
