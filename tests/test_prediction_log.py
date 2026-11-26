import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, create_engine, select

from hotel_booking import config, data, prediction_log, replay, sample_model
from hotel_booking.api import create_app
from hotel_booking.prediction_log import labels, prediction_logs

SPEC = {"name": "Logistic Regression", "params": {"C": 0.1}}
SCHEMA_SQL = Path(__file__).resolve().parents[1] / "docs" / "examples" / "sql" / "schema.sql"


@pytest.fixture(scope="module")
def model_dir(sample: pd.DataFrame, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small model saved for the API."""
    splits = data.time_split(sample)
    model, card = sample_model.train_with_card(SPEC, splits["train"], splits["test"], "LogReg", "log-test")
    folder = tmp_path_factory.mktemp("model")
    sample_model.save_artifacts(model, card, folder)
    return folder


@pytest.fixture
def database_url(tmp_path: Path) -> str:
    """A fresh SQLite database standing in for PostgreSQL."""
    return f"sqlite:///{(tmp_path / 'log.db').as_posix()}"


@pytest.fixture
def client(model_dir: Path, database_url: str) -> Iterator[TestClient]:
    """API writing its predictions to the test database."""
    app = create_app(model_dir / "model.joblib", model_dir / "model_card.json", database_url=database_url)
    with TestClient(app) as test_client:
        yield test_client


def rows(database_url: str) -> list[dict[str, Any]]:
    """All prediction log rows, oldest first."""
    engine = create_engine(database_url)
    with engine.connect() as connection:
        result = [dict(row._mapping) for row in connection.execute(select(prediction_logs).order_by("id"))]
    engine.dispose()
    return result


def test_predict_writes_one_log_row(client: TestClient, database_url: str, booking: dict[str, Any]) -> None:
    """A prediction is logged with its input, probability, decision, version and timing."""
    body = client.post("/predict", json={**booking, "booking_id": 42}, headers={"X-Replay-Month": "2017-03"}).json()
    assert client.get("/health").json()["prediction_logging"] is True
    [row] = rows(database_url)
    assert row["endpoint"] == "/predict" and row["model_version"] == "log-test" and row["backend"] == "joblib"
    assert row["booking_id"] == 42 and row["replay_month"] == "2017-03" and row["row_index"] == 0
    assert row["probability"] == pytest.approx(body["probability"]) and row["is_canceled"] == body["is_canceled"]
    assert row["threshold"] == pytest.approx(body["threshold"]) and row["latency_ms"] > 0
    assert set(row["input"]) == set(config.MODEL_INPUT_COLUMNS) and "booking_id" not in row["input"]
    assert row["warnings"] == []


def test_batch_writes_one_row_per_booking(client: TestClient, database_url: str, booking: dict[str, Any]) -> None:
    """A batch shares one request id, keeps the booking order and stores the filled rooms_in_booking."""
    plain = {**booking, "rooms_in_booking": None}
    bookings = [{**plain, "booking_id": 1}, {**plain, "booking_id": 2}, {**booking, "hotel": "Airport Hotel"}]
    client.post("/predict/batch", json={"bookings": bookings})
    logged = rows(database_url)
    assert [row["row_index"] for row in logged] == [0, 1, 2]
    assert len({row["request_id"] for row in logged}) == 1
    assert [row["booking_id"] for row in logged] == [1, 2, None]
    assert [row["input"]["rooms_in_booking"] for row in logged[:2]] == [2, 2]
    assert {"column": "hotel", "issue": "unseen category", "rows": 1} in logged[0]["warnings"]


def test_booking_id_does_not_change_the_prediction(client: TestClient, booking: dict[str, Any]) -> None:
    """The log-only booking id never reaches the model."""
    with_id = client.post("/predict", json={**booking, "booking_id": 7}).json()["probability"]
    assert client.post("/predict", json=booking).json()["probability"] == pytest.approx(with_id)


def test_replay_month_header_is_validated(client: TestClient, booking: dict[str, Any]) -> None:
    """X-Replay-Month must look like YYYY-MM."""
    assert client.post("/predict", json=booking, headers={"X-Replay-Month": "March"}).status_code == 422


def test_without_database_nothing_is_logged(model_dir: Path, booking: dict[str, Any]) -> None:
    """With no database URL the API serves predictions and reports logging as off."""
    app = create_app(model_dir / "model.joblib", model_dir / "model_card.json", database_url="")
    with TestClient(app) as no_log:
        assert no_log.post("/predict", json=booking).status_code == 200
        assert no_log.get("/health").json()["prediction_logging"] is False


def test_failed_writes_are_counted_not_raised(
    client: TestClient, database_url: str, booking: dict[str, Any], tmp_path: Path
) -> None:
    """When the database breaks, predictions still succeed and the failure is counted."""
    engine: Engine = client.app.state.prediction_log.engine  # type: ignore[attr-defined]
    with engine.begin() as connection:
        connection.exec_driver_sql("DROP TABLE prediction_logs")
    assert client.post("/predict", json=booking).status_code == 200
    assert "api_prediction_log_errors_total 1.0" in client.get("/metrics").text


def test_unreachable_database_disables_the_log(model_dir: Path, booking: dict[str, Any], tmp_path: Path) -> None:
    """A database that cannot be opened at startup disables the log instead of the API."""
    bad_url = f"sqlite:///{(tmp_path / 'missing' / 'dir' / 'log.db').as_posix()}"
    with TestClient(create_app(model_dir / "model.joblib", model_dir / "model_card.json", database_url=bad_url)) as api:
        assert api.post("/predict", json=booking).status_code == 200
        assert api.get("/health").json()["prediction_logging"] is False
        assert "api_prediction_log_errors_total 1.0" in api.get("/metrics").text


def test_replay_logs_ids_and_months_and_releases_labels_to_the_database(
    model_dir: Path, database_url: str, raw_bookings: pd.DataFrame, tmp_path: Path
) -> None:
    """Replaying a month logs every booking with its id and month, then upserts its labels."""
    (tmp_path / "raw").mkdir()
    raw_bookings.to_csv(tmp_path / "raw" / config.RAW_FILE, index=False)
    data.preprocess(tmp_path / "raw", tmp_path / "processed")
    batch_file = replay.build_stream(tmp_path / "processed", tmp_path / "stream")[0]
    engine = prediction_log.connect(database_url)
    app = create_app(model_dir / "model.joblib", model_dir / "model_card.json", database_url=database_url)
    with TestClient(app) as api:
        for _ in range(2):
            summary = replay.replay_month(api, batch_file, tmp_path / "labels", 5, 0, engine)
    month = data.read_split(batch_file.parent, batch_file.stem)
    logged = rows(database_url)
    assert len(logged) == 2 * len(month) and {row["replay_month"] for row in logged} == {batch_file.stem}
    assert {row["booking_id"] for row in logged} == set(month[config.ID_COLUMN])
    with engine.connect() as connection:
        released = pd.DataFrame(connection.execute(select(labels)).mappings().all())
    engine.dispose()
    assert summary["labels_in_db"] == len(month) == len(released)
    expected = dict(zip(month[config.ID_COLUMN], month[config.TARGET], strict=True))
    assert dict(zip(released["booking_id"], released["is_canceled"], strict=True)) == expected


def test_tables_match_the_documented_schema() -> None:
    """The SQLAlchemy tables have the same columns as docs/examples/sql/schema.sql."""
    sql = SCHEMA_SQL.read_text(encoding="utf-8")
    for table in (prediction_logs, labels):
        body = re.search(rf"CREATE TABLE IF NOT EXISTS {table.name} \((.*?)\n\);", sql, re.S)
        assert body is not None
        documented = [line.split()[0] for line in body.group(1).strip().splitlines() if line.strip()]
        assert documented == [column.name for column in table.columns]
