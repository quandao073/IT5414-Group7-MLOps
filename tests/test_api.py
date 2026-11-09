from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from hotel_booking import config, data, sample_model
from hotel_booking.api import create_app
from hotel_booking.schemas import Booking

SPEC = {"name": "Logistic Regression", "half_life_days": 365, "pipeline": {"log_skewed": True}, "params": {"C": 0.1}}
VERSION = "test-logreg-1"


@pytest.fixture(scope="module")
def model_dir(sample: pd.DataFrame, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small model trained on synthetic bookings, saved with its card as the API expects."""
    splits = data.time_split(sample)
    model, card = sample_model.train_with_card(SPEC, splits["train"], splits["test"], "Logistic Regression", VERSION)
    folder = tmp_path_factory.mktemp("models")
    sample_model.save_artifacts(model, card, folder)
    return folder


@pytest.fixture(scope="module")
def client(model_dir: Path) -> Iterator[TestClient]:
    """API client serving the synthetic model."""
    app = create_app(model_dir / sample_model.MODEL_FILE, model_dir / sample_model.CARD_FILE)
    with TestClient(app) as test_client:
        yield test_client


def test_booking_schema_matches_model_inputs() -> None:
    """The request schema has exactly the columns the model pipeline reads."""
    assert list(Booking.model_fields) == config.MODEL_INPUT_COLUMNS


def test_health_reports_loaded_model(client: TestClient) -> None:
    """Health shows the loaded model version."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True, "model_version": VERSION}


def test_predict_returns_probability_and_decision(client: TestClient, booking: dict[str, Any]) -> None:
    """A valid booking gets a probability, the decision at the card threshold and the model version."""
    body = client.post("/predict", json=booking).json()
    assert 0 <= body["probability"] <= 1
    assert body["is_canceled"] == (body["probability"] >= body["threshold"])
    assert body["model_version"] == VERSION


def test_predict_defaults_rooms_in_booking_to_one(client: TestClient, booking: dict[str, Any]) -> None:
    """Omitting rooms_in_booking for a single booking is the same as sending 1."""
    without = client.post("/predict", json={**booking, "rooms_in_booking": None}).json()
    explicit = client.post("/predict", json={**booking, "rooms_in_booking": 1}).json()
    assert without["probability"] == pytest.approx(explicit["probability"])


@pytest.mark.parametrize(
    "change",
    [
        {"lead_time": "abc"},
        {"lead_time": -1},
        {"arrival_date_day_of_month": 32},
        {"is_repeated_guest": 2},
        {"rooms_in_booking": 0},
        {"unknown_column": 1},
    ],
)
def test_predict_rejects_malformed_input(client: TestClient, booking: dict[str, Any], change: dict[str, Any]) -> None:
    """Wrong types, impossible values and unknown fields are rejected with 422."""
    assert client.post("/predict", json={**booking, **change}).status_code == 422


def test_predict_rejects_missing_column(client: TestClient, booking: dict[str, Any]) -> None:
    """A booking without a required column is rejected with 422 naming the column."""
    response = client.post("/predict", json={k: v for k, v in booking.items() if k != "hotel"})
    assert response.status_code == 422
    assert any(error["loc"][-1] == "hotel" for error in response.json()["detail"])


def test_predict_warns_about_unseen_values(client: TestClient, booking: dict[str, Any]) -> None:
    """An unseen category is still predicted and reported as a warning."""
    response = client.post("/predict", json={**booking, "hotel": "Airport Hotel"})
    assert response.status_code == 200
    assert {"column": "hotel", "issue": "unseen category", "rows": 1} in response.json()["warnings"]


def test_batch_keeps_request_order(client: TestClient, booking: dict[str, Any]) -> None:
    """Batch predictions match single predictions, in request order."""
    bookings = [booking, {**booking, "lead_time": 300, "deposit_type": "Non Refund"}]
    batch = client.post("/predict/batch", json={"bookings": bookings}).json()
    singles = [client.post("/predict", json=item).json()["probability"] for item in bookings]
    assert [item["probability"] for item in batch["predictions"]] == pytest.approx(singles)
    assert batch["model_version"] == VERSION


def test_batch_counts_rooms_of_identical_bookings(client: TestClient, booking: dict[str, Any]) -> None:
    """Identical bookings without rooms_in_booking are counted as one group booking."""
    plain = {**booking, "rooms_in_booking": None}
    counted = client.post("/predict/batch", json={"bookings": [plain, plain]}).json()
    explicit = client.post("/predict", json={**booking, "rooms_in_booking": 2}).json()
    assert counted["predictions"][0]["probability"] == pytest.approx(explicit["probability"])


def test_batch_rejects_empty_list(client: TestClient) -> None:
    """An empty batch is rejected with 422."""
    assert client.post("/predict/batch", json={"bookings": []}).status_code == 422


def test_without_model_health_is_up_and_predict_unavailable(tmp_path: Path, booking: dict[str, Any]) -> None:
    """Without a model file the API still starts, reports no model and answers predictions with 503."""
    app = create_app(tmp_path / "missing.joblib", tmp_path / "missing.json")
    with TestClient(app) as no_model:
        assert no_model.get("/health").json()["model_loaded"] is False
        assert no_model.post("/predict", json=booking).status_code == 503
