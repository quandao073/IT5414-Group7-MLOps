import json

import httpx
import numpy as np
import pandas as pd
import pytest

from hotel_booking.api_client import ApiClient, ApiError, csv_to_bookings
from hotel_booking.config import MODEL_INPUT_COLUMNS


def test_csv_to_bookings_drops_columns_the_api_rejects(booking: dict) -> None:
    frame = pd.DataFrame([{**booking, "booking_id": 7, "is_canceled": 1, "assigned_room_type": "A"}])
    payloads, missing = csv_to_bookings(frame)
    assert missing == []
    assert set(payloads[0]) == set(MODEL_INPUT_COLUMNS)


def test_csv_to_bookings_turns_nan_into_none(booking: dict) -> None:
    frame = pd.DataFrame([{**booking, "children": np.nan, "agent": np.nan, "company": np.nan}])
    payloads, _ = csv_to_bookings(frame)
    assert payloads[0]["children"] is None
    assert payloads[0]["agent"] is None
    assert payloads[0]["company"] is None


def test_csv_to_bookings_reports_missing_columns(booking: dict) -> None:
    frame = pd.DataFrame([booking]).drop(columns=["adr", "hotel"])
    payloads, missing = csv_to_bookings(frame)
    assert missing == ["hotel", "adr"]
    assert payloads == []


def test_payload_is_json_serialisable(booking: dict) -> None:
    frame = pd.DataFrame([booking])
    payloads, _ = csv_to_bookings(frame)
    assert json.loads(json.dumps(payloads[0]))["adults"] == 2


def make_client(handler, base_url: str = "http://test") -> ApiClient:
    return ApiClient(base_url, timeout=1.0, transport=httpx.MockTransport(handler))


def test_health_returns_model_version() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/health"
        return httpx.Response(200, json={"status": "ok", "model_loaded": True, "model_version": "v1"})

    health = make_client(handler).health()
    assert health.model_loaded is True
    assert health.model_version == "v1"


def test_predict_one_returns_probability(booking: dict) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/predict"
        assert json.loads(request.content)["hotel"] == "City Hotel"
        return httpx.Response(
            200,
            json={"probability": 0.25, "is_canceled": True, "threshold": 0.18, "model_version": "v1", "warnings": []},
        )

    result = make_client(handler).predict_one(booking)
    assert result.prediction.probability == 0.25
    assert result.prediction.is_canceled is True
    assert result.threshold == 0.18
    assert result.model_version == "v1"


def test_predict_batch_preserves_order(booking: dict) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/predict/batch"
        assert len(json.loads(request.content)["bookings"]) == 2
        return httpx.Response(
            200,
            json={
                "predictions": [{"probability": 0.9, "is_canceled": True}, {"probability": 0.1, "is_canceled": False}],
                "threshold": 0.18,
                "model_version": "v1",
                "warnings": [],
            },
        )

    result = make_client(handler).predict_batch([booking, booking])
    assert [p.probability for p in result.predictions] == [0.9, 0.1]


def test_predict_batch_rejects_more_than_max_batch_size(booking: dict) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    with pytest.raises(ApiError, match="10.000"):
        make_client(handler).predict_batch([booking] * 10_001)


def test_predict_one_maps_connection_error(booking: dict) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ApiError, match="Không kết nối được API tới"):
        make_client(handler).predict_one(booking)


def test_predict_one_maps_timeout(booking: dict) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ApiError, match="không phản hồi"):
        make_client(handler).predict_one(booking)


def test_predict_one_maps_422_to_column_names(booking: dict) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={"detail": [{"loc": ["body", "adr"], "msg": "bad"}, {"loc": ["body", "adults"], "msg": "bad"}]},
        )

    with pytest.raises(ApiError) as excinfo:
        make_client(handler).predict_one(booking)
    assert "adr" in str(excinfo.value)
    assert "adults" in str(excinfo.value)


def test_predict_one_maps_503(booking: dict) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "No model loaded."})

    with pytest.raises(ApiError, match="chưa được nhập"):
        make_client(handler).predict_one(booking)
