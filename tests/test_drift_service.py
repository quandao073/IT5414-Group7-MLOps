import json
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from hotel_booking import config, data, drift_service

LOG_TABLE = """
CREATE TABLE prediction_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TIMESTAMP NOT NULL,
    replay_month TEXT,
    input TEXT NOT NULL
)
"""
START = datetime(2026, 11, 1, 12, 0, 0)


def write_logs(url: str, frames: dict[str, pd.DataFrame]) -> None:
    """Log every row of each frame under its replay month, one second apart, as the API would."""
    engine = create_engine(url)
    rows: list[dict[str, object]] = []
    for month, frame in frames.items():
        for record in frame[config.MODEL_INPUT_COLUMNS].to_dict(orient="records"):
            clean = {key: None if pd.isna(value) else value for key, value in record.items()}
            rows.append(
                {"created_at": START + timedelta(seconds=len(rows)), "month": month, "input": json.dumps(clean)}
            )
    with engine.begin() as connection:
        connection.execute(text(LOG_TABLE))
        connection.execute(
            text("INSERT INTO prediction_logs (created_at, replay_month, input) VALUES (:created_at, :month, :input)"),
            rows,
        )
    engine.dispose()


@pytest.fixture
def setup(sample: pd.DataFrame, tmp_path: Path) -> tuple[str, Path]:
    """A log database with an unchanged month and a shifted month, plus the processed training split."""
    splits = data.time_split(sample)
    processed = tmp_path / "processed"
    processed.mkdir()
    for name, frame in splits.items():
        frame.to_csv(processed / f"{name}.csv", index=False)
    train = splits["train"]
    shifted = train.assign(lead_time=train["lead_time"] + 500, adr=train["adr"] * 4)
    url = f"sqlite:///{(tmp_path / 'logs.db').as_posix()}"
    write_logs(url, {"2017-01": train, "2017-02": shifted})
    return url, processed


@pytest.fixture
def client(setup: tuple[str, Path]) -> Iterator[TestClient]:
    """The drift service on the test database."""
    url, processed = setup
    with TestClient(drift_service.create_app(url, processed, min_rows=50)) as test_client:
        yield test_client


def test_health_reports_reference_and_database(client: TestClient) -> None:
    """Health shows the reference size and that a database is configured."""
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["database"] is True and body["reference_rows"] > 0


def test_unchanged_month_does_not_drift_and_shifted_month_does(client: TestClient) -> None:
    """Logging the training data itself shows no drift; shifting lead_time and adr makes both drift."""
    same = client.get("/drift", params={"month": "2017-01"}).json()
    assert same["drifted_columns"] == 0 and not same["dataset_drift"]
    shifted = client.get("/drift", params={"month": "2017-02"}).json()
    drifted = {row["column"] for row in shifted["details"] if row["drifted"]}
    assert {"lead_time", "adr"} <= drifted
    assert shifted["drift_share"] == round(len(drifted) / shifted["columns"], 4)


def test_latest_uses_most_recent_month(client: TestClient) -> None:
    """`/drift/latest` checks the latest replay month."""
    assert client.get("/drift/latest").json()["window"] == "2017-02"


def test_time_window_and_bad_parameters(client: TestClient, setup: tuple[str, Path]) -> None:
    """A created_at range selects rows by time; mixing or missing parameters is rejected."""
    total = len(drift_service.load_logged_inputs(create_engine(setup[0]), month="2017-01"))
    end = START + timedelta(seconds=total)
    body = client.get("/drift", params={"start": START.isoformat(), "end": end.isoformat()}).json()
    assert body["rows"] == total
    assert client.get("/drift").status_code == 422
    assert client.get("/drift", params={"month": "2017-01", "start": START.isoformat()}).status_code == 422


def test_too_few_rows_is_not_found(client: TestClient) -> None:
    """A window with fewer logged rows than the minimum is refused instead of reporting noise."""
    response = client.get("/drift", params={"month": "2017-12"})
    assert response.status_code == 404 and "0 logged rows" in response.json()["detail"]


def test_metrics_expose_latest_result(client: TestClient) -> None:
    """After a check, /metrics exposes the drift share of that month."""
    share = client.get("/drift", params={"month": "2017-02"}).json()["drift_share"]
    metrics = client.get("/metrics").text
    assert f'drift_share{{window="2017-02"}} {share}' in metrics
    assert "drift_checks_total 1.0" in metrics


def test_report_returns_html(client: TestClient) -> None:
    """The Evidently report of a month is served as HTML."""
    response = client.get("/drift/report", params={"month": "2017-02"})
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/html")


def test_without_database_drift_is_unavailable(setup: tuple[str, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    """Without DATABASE_URL the service starts but drift endpoints answer 503."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with TestClient(drift_service.create_app(reference_dir=setup[1])) as test_client:
        assert test_client.get("/health").json()["database"] is False
        assert test_client.get("/drift/latest").status_code == 503
