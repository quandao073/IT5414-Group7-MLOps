"""Prediction log and delayed labels in a SQL database (contract C9 in `docs/source/contracts.md`).

The tables mirror `docs/examples/sql/schema.sql`. They are defined with SQLAlchemy so the same code writes to
PostgreSQL in production and to SQLite in tests; `JSONB` and `UUID` fall back to portable types outside PostgreSQL.
"""

import json
import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Engine,
    Float,
    Index,
    Integer,
    MetaData,
    SmallInteger,
    Table,
    Text,
    Uuid,
    create_engine,
    func,
)
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.dialects.postgresql import JSONB

logger = logging.getLogger(__name__)
metadata = MetaData()
JsonColumn = JSON().with_variant(JSONB(), "postgresql")

prediction_logs = Table(
    "prediction_logs",
    metadata,
    Column("id", BigInteger().with_variant(Integer(), "sqlite"), primary_key=True, autoincrement=True),
    Column("request_id", Uuid(), nullable=False),
    Column("row_index", Integer(), nullable=False, server_default="0"),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("endpoint", Text(), nullable=False),
    Column("model_version", Text(), nullable=False),
    Column("backend", Text(), nullable=False, server_default="joblib"),
    Column("booking_id", BigInteger()),
    Column("replay_month", Text()),
    Column("input", JsonColumn, nullable=False),
    Column("probability", Float(), nullable=False),
    Column("threshold", Float(), nullable=False),
    Column("is_canceled", Boolean(), nullable=False),
    Column("latency_ms", Float(), nullable=False),
    Column("warnings", JsonColumn, nullable=False),
    Index("prediction_logs_created_at_idx", "created_at"),
    Index("prediction_logs_replay_month_idx", "replay_month"),
    Index("prediction_logs_model_version_idx", "model_version"),
    Index("prediction_logs_booking_id_idx", "booking_id"),
)

labels = Table(
    "labels",
    metadata,
    Column("booking_id", BigInteger(), primary_key=True, autoincrement=False),
    Column("is_canceled", SmallInteger(), nullable=False),
    Column("month", Text(), nullable=False),
    Column("released_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Index("labels_month_idx", "month"),
)


def connect(database_url: str) -> Engine:
    """Create an engine and the tables that do not exist yet.

    Args:
        database_url: SQLAlchemy URL, e.g. `postgresql+psycopg://user:password@postgres:5432/hotel`.

    Returns:
        The engine.
    """
    # Fail fast when PostgreSQL is unreachable, so the API still starts and serves without the log.
    connect_args = {"connect_timeout": 3} if database_url.startswith("postgresql") else {}
    engine = create_engine(database_url, pool_pre_ping=True, connect_args=connect_args)
    metadata.create_all(engine, checkfirst=True)
    return engine


@dataclass(frozen=True)
class PredictionRecord:
    """Everything logged for one request."""

    endpoint: str
    model_version: str
    backend: str
    threshold: float
    latency_ms: float
    inputs: pd.DataFrame
    probabilities: Sequence[float]
    booking_ids: Sequence[int | None]
    replay_month: str | None
    warnings: list[dict[str, Any]]


def to_rows(record: PredictionRecord) -> list[dict[str, Any]]:
    """One row per scored booking, sharing the request id, latency and warnings.

    Args:
        record: The logged request.

    Returns:
        Rows for `prediction_logs`.
    """
    request_id = uuid.uuid4()
    inputs = json.loads(record.inputs.to_json(orient="records"))
    return [
        {
            "request_id": request_id,
            "row_index": index,
            "endpoint": record.endpoint,
            "model_version": record.model_version,
            "backend": record.backend,
            "booking_id": booking_id,
            "replay_month": record.replay_month,
            "input": values,
            "probability": float(probability),
            "threshold": record.threshold,
            "is_canceled": bool(probability >= record.threshold),
            "latency_ms": record.latency_ms,
            "warnings": record.warnings,
        }
        for index, (values, probability, booking_id) in enumerate(
            zip(inputs, record.probabilities, record.booking_ids, strict=True)
        )
    ]


class PredictionLog:
    """Writes prediction rows; failures are logged and counted, never raised into the request."""

    def __init__(self, engine: Engine, on_error: Any = None) -> None:
        """Wrap an engine.

        Args:
            engine: Engine from `connect`.
            on_error: Called with the exception when a write fails, e.g. to count it in metrics.
        """
        self.engine = engine
        self.on_error = on_error

    def write(self, record: PredictionRecord) -> None:
        """Insert the rows of one request.

        Args:
            record: The logged request.
        """
        try:
            with self.engine.begin() as connection:
                connection.execute(prediction_logs.insert(), to_rows(record))
        except Exception as error:
            logger.warning("Could not write %d prediction rows: %s", len(record.probabilities), error)
            if self.on_error is not None:
                self.on_error(error)


def write_labels(engine: Engine, batch: pd.DataFrame, month: str, id_column: str, target: str) -> int:
    """Insert or update the released labels of a month.

    Args:
        engine: Engine from `connect`.
        batch: Bookings of the month with their id and label.
        month: The month, `YYYY-MM`.
        id_column: Name of the booking id column.
        target: Name of the label column.

    Returns:
        The number of labels written.
    """
    released_at = datetime.now(UTC)
    rows = [
        {"booking_id": int(booking), "is_canceled": int(label), "month": month, "released_at": released_at}
        for booking, label in zip(batch[id_column], batch[target], strict=True)
    ]
    if not rows:
        return 0
    dialect = postgresql if engine.dialect.name == "postgresql" else sqlite
    statement = dialect.insert(labels)
    statement = statement.on_conflict_do_update(
        index_elements=["booking_id"],
        set_={
            "is_canceled": statement.excluded.is_canceled,
            "month": statement.excluded.month,
            "released_at": statement.excluded.released_at,
        },
    )
    with engine.begin() as connection:
        connection.execute(statement, rows)
    return len(rows)
