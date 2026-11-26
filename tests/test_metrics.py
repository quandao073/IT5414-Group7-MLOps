from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from hotel_booking import data, sample_model
from hotel_booking.api import create_app
from hotel_booking.monitoring import error_type

SPEC = {"name": "Logistic Regression", "params": {"C": 0.1}}
VERSION = "metrics-test-1"


@pytest.fixture
def client(sample: pd.DataFrame, tmp_path: Path) -> Iterator[TestClient]:
    """A fresh API with its own metrics, serving a small synthetic model."""
    splits = data.time_split(sample)
    model, card = sample_model.train_with_card(SPEC, splits["train"], splits["test"], "LogReg", VERSION)
    sample_model.save_artifacts(model, card, tmp_path)
    app = create_app(tmp_path / sample_model.MODEL_FILE, tmp_path / sample_model.CARD_FILE)
    with TestClient(app) as test_client:
        yield test_client


def scrape(client: TestClient) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
    """Read `/metrics` into {(sample name, sorted labels): value}."""
    response = client.get("/metrics")
    assert response.status_code == 200 and response.headers["content-type"].startswith("text/plain")
    return {
        (sample.name, tuple(sorted(sample.labels.items()))): sample.value
        for family in text_string_to_metric_families(response.text)
        for sample in family.samples
    }


def value(samples: dict[tuple[str, tuple[tuple[str, str], ...]], float], name: str, **labels: str) -> float:
    """One sample value, 0 when absent."""
    return samples.get((name, tuple(sorted(labels.items()))), 0.0)


def test_metrics_expose_the_served_model(client: TestClient) -> None:
    """The model info gauge names the loaded version and backend."""
    samples = scrape(client)
    assert value(samples, "api_model_info", model_version=VERSION, backend="joblib") == 1


def test_requests_and_latency_are_counted_per_route(client: TestClient, booking: dict[str, Any]) -> None:
    """Requests are counted by route template and status, with one latency observation each."""
    for _ in range(3):
        client.post("/predict", json=booking)
    client.post("/predict", json={**booking, "lead_time": "abc"})
    client.get("/does-not-exist")
    samples = scrape(client)
    assert value(samples, "api_requests_total", endpoint="/predict", status="200") == 3
    assert value(samples, "api_requests_total", endpoint="/predict", status="422") == 1
    assert value(samples, "api_requests_total", endpoint="unmatched", status="404") == 1
    assert value(samples, "api_request_latency_seconds_count", endpoint="/predict") == 4
    assert value(samples, "api_errors_total", endpoint="/predict", type="validation") == 1


def test_predictions_feed_probability_and_decision_metrics(client: TestClient, booking: dict[str, Any]) -> None:
    """Every scored booking is one probability observation and one decision."""
    bookings = [booking, {**booking, "lead_time": 300, "deposit_type": "Non Refund"}, booking]
    client.post("/predict/batch", json={"bookings": bookings})
    client.post("/predict", json=booking)
    samples = scrape(client)
    assert value(samples, "api_prediction_probability_count") == 4
    decided = value(samples, "api_predictions_total", outcome="canceled") + value(
        samples, "api_predictions_total", outcome="not_canceled"
    )
    assert decided == 4


def test_input_warnings_are_counted(client: TestClient, booking: dict[str, Any]) -> None:
    """Unseen categories are counted per column and issue."""
    client.post("/predict/batch", json={"bookings": [{**booking, "hotel": "Airport Hotel"}] * 2})
    samples = scrape(client)
    assert value(samples, "api_input_warnings_total", column="hotel", issue="unseen category") == 2


def test_missing_model_is_reported_as_unavailable(tmp_path: Path, booking: dict[str, Any]) -> None:
    """Without a model, no model info is exposed and predictions count as unavailable errors."""
    with TestClient(create_app(tmp_path / "none.joblib", tmp_path / "none.json")) as no_model:
        no_model.post("/predict", json=booking)
        samples = scrape(no_model)
    assert not any(name == "api_model_info" for name, _ in samples)
    assert value(samples, "api_errors_total", endpoint="/predict", type="unavailable") == 1


@pytest.mark.parametrize(
    "status, expected", [(200, None), (404, "client"), (422, "validation"), (500, "internal"), (503, "unavailable")]
)
def test_error_type(status: int, expected: str | None) -> None:
    """Status codes map to error types."""
    assert error_type(status) == expected
