"""Prediction API: serves the model saved by `sample_model.py` (contract C1 in `docs/contracts.md`).

Run locally with `uv run invoke serve-api` and open http://localhost:8000/docs.
"""

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from hotel_booking.config import GROUP_SIZE_COLUMN, MODEL_INPUT_COLUMNS
from hotel_booking.data import add_group_size, validate_inputs
from hotel_booking.sample_model import CARD_FILE, MODEL_FILE
from hotel_booking.schemas import (
    BatchRequest,
    BatchResponse,
    Booking,
    HealthResponse,
    InputWarning,
    Prediction,
    PredictResponse,
)

NULLABLE_FLOATS = ["children", "agent", "company"]


@dataclass
class ServedModel:
    """A fitted model with the parts of its card the API needs."""

    model: Any
    version: str
    threshold: float
    schema: dict[str, Any]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Cancellation probabilities.

        Args:
            frame: Bookings with the columns in `MODEL_INPUT_COLUMNS`.

        Returns:
            One probability per row.
        """
        return self.model.predict_proba(frame[MODEL_INPUT_COLUMNS])[:, 1]


def load_served_model(model_path: Path, card_path: Path) -> ServedModel:
    """Load a joblib model and its card.

    Args:
        model_path: Path to `model.joblib`.
        card_path: Path to `model_card.json`.

    Returns:
        The model ready to serve.
    """
    card = json.loads(card_path.read_text(encoding="utf-8"))
    return ServedModel(
        model=joblib.load(model_path),
        version=str(card["model_version"]),
        threshold=float(card["decision_threshold"]),
        schema=card["input_schema"],
    )


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


def create_app(model_path: Path | None = None, card_path: Path | None = None) -> FastAPI:
    """Build the API.

    Args:
        model_path: Model file; defaults to the `MODEL_PATH` environment variable, then `models/model.joblib`.
        card_path: Model card; defaults to `MODEL_CARD_PATH`, then `models/model_card.json`.

    Returns:
        The FastAPI application.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        model_file = model_path or Path(os.getenv("MODEL_PATH", f"models/{MODEL_FILE}"))
        card_file = card_path or Path(os.getenv("MODEL_CARD_PATH", f"models/{CARD_FILE}"))
        app.state.served = None
        if model_file.exists() and card_file.exists():
            app.state.served = load_served_model(model_file, card_file)
        yield

    app = FastAPI(
        title="Hotel booking cancellation API",
        description="Predicts the probability that a hotel booking is canceled.",
        version="0.1.0",
        lifespan=lifespan,
    )

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
        return PredictResponse(
            probability=probability,
            is_canceled=probability >= served.threshold,
            threshold=served.threshold,
            model_version=served.version,
            warnings=input_warnings(frame, served.schema),
        )

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(batch: BatchRequest, request: Request) -> BatchResponse:
        """Predict the cancellation probabilities of several bookings, in request order."""
        served = served_model(request)
        frame = bookings_to_frame(batch.bookings)
        probabilities = served.predict(frame)
        return BatchResponse(
            predictions=[
                Prediction(probability=float(value), is_canceled=bool(value >= served.threshold))
                for value in probabilities
            ],
            threshold=served.threshold,
            model_version=served.version,
            warnings=input_warnings(frame, served.schema),
        )

    return app


app = create_app()
