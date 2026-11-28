"""Locust load test for POST /predict; run the API with MODEL_BACKEND=joblib or onnx."""

import json
from pathlib import Path

from locust import HttpUser, between, task

ROOT = Path(__file__).resolve().parents[2]
BOOKING = json.loads((ROOT / "docs" / "examples" / "predict_request.json").read_text(encoding="utf-8"))


class BookingUser(HttpUser):
    """Posts one booking at a time and checks the prediction response."""

    wait_time = between(0.01, 0.05)

    @task
    def predict(self) -> None:
        with self.client.post("/predict", json=BOOKING, catch_response=True, name="/predict") as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return
            try:
                body = response.json()
            except ValueError:
                response.failure("response is not JSON")
                return
            if "probability" not in body:
                response.failure("response has no probability")
