import json
from pathlib import Path

from locust import HttpUser
from locustfile import BookingUser

from hotel_booking.schemas import Booking

ROOT = Path(__file__).resolve().parents[2]


def test_example_payload_is_a_valid_booking() -> None:
    payload = json.loads((ROOT / "docs" / "examples" / "predict_request.json").read_text(encoding="utf-8"))
    assert Booking.model_validate(payload).hotel == "City Hotel"


def test_user_posts_predict() -> None:
    assert issubclass(BookingUser, HttpUser)
    assert any(getattr(task, "__name__", "") == "predict" for task in BookingUser.tasks)
