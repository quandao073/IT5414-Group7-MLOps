import json

import httpx
import numpy as np
import pandas as pd
import pytest

from hotel_booking.api_client import ApiClient, ApiError, csv_to_bookings
from hotel_booking.config import MODEL_INPUT_COLUMNS
from hotel_booking.schemas import MAX_BATCH_SIZE


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


def test_csv_to_bookings_turns_pd_na_into_none(booking: dict) -> None:
    frame = pd.DataFrame([{**booking, "children": pd.NA, "agent": pd.NA, "company": pd.NA}])
    payloads, _ = csv_to_bookings(frame)
    assert payloads[0]["children"] is None
    assert payloads[0]["agent"] is None
    assert payloads[0]["company"] is None
    assert json.loads(json.dumps(payloads[0]))["children"] is None


def test_csv_to_bookings_turns_blank_rooms_in_booking_into_none(booking: dict) -> None:
    frame = pd.DataFrame([{**booking, "rooms_in_booking": np.nan}])
    payloads, _ = csv_to_bookings(frame)
    assert payloads[0]["rooms_in_booking"] is None
    assert json.loads(json.dumps(payloads[0]))["rooms_in_booking"] is None


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


def test_predict_batch_accepts_exactly_max_batch_size(booking: dict) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert len(json.loads(request.content)["bookings"]) == MAX_BATCH_SIZE
        return httpx.Response(
            200,
            json={
                "predictions": [{"probability": 0.5, "is_canceled": False}],
                "threshold": 0.18,
                "model_version": "v1",
                "warnings": [],
            },
        )

    make_client(handler).predict_batch([booking] * MAX_BATCH_SIZE)
    assert len(requests) == 1


def test_predict_batch_rejects_more_than_max_batch_size(booking: dict) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    with pytest.raises(ApiError) as excinfo:
        make_client(handler).predict_batch([booking] * (MAX_BATCH_SIZE + 1))
    assert str(excinfo.value) == "Chỉ chấm tối đa 10.000 bản ghi mỗi lần."


def test_predict_one_maps_connection_error(booking: dict) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ApiError) as excinfo:
        make_client(handler).predict_one(booking)
    assert str(excinfo.value) == ("Không kết nối được API tại http://test. Chạy `uv run invoke serve-api` để bật API.")


def test_predict_one_maps_timeout(booking: dict) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ApiError) as excinfo:
        make_client(handler).predict_one(booking)
    assert str(excinfo.value) == "API tại http://test không phản hồi sau 1 giây."


def test_predict_one_maps_422_to_column_names(booking: dict) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={"detail": [{"loc": ["body", "adr"], "msg": "bad"}, {"loc": ["body", "adults"], "msg": "bad"}]},
        )

    with pytest.raises(ApiError) as excinfo:
        make_client(handler).predict_one(booking)
    assert str(excinfo.value) == "Dữ liệu không hợp lệ (422) ở cột: adr, adults"


def test_predict_batch_422_names_the_invalid_columns(booking: dict) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            422,
            json={
                "detail": [
                    {"loc": ["body", "bookings", 0, "adr"], "msg": "bad"},
                    {"loc": ["body", "bookings", 1, "adr"], "msg": "bad"},
                    {"loc": ["body", "bookings", 1, "adults"], "msg": "bad"},
                ]
            },
        )

    with pytest.raises(ApiError) as excinfo:
        make_client(handler).predict_batch([booking, booking])
    assert str(excinfo.value) == "Dữ liệu không hợp lệ (422) ở cột: adr, adults"


def test_predict_batch_422_caps_the_column_list(booking: dict) -> None:
    detail = [{"loc": ["body", "bookings", 0, f"col{index}"], "msg": "bad"} for index in range(12)]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": detail})

    with pytest.raises(ApiError) as excinfo:
        make_client(handler).predict_batch([booking])
    message = str(excinfo.value)
    assert message == "Dữ liệu không hợp lệ (422) ở cột: " + ", ".join(f"col{index}" for index in range(10)) + " …"


def test_predict_batch_422_without_a_detail_list_falls_back(booking: dict) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "not a list"})

    with pytest.raises(ApiError) as excinfo:
        make_client(handler).predict_batch([booking])
    assert str(excinfo.value).startswith("Dữ liệu không hợp lệ (422): ")


def test_predict_batch_422_with_a_non_dict_item_falls_back(booking: dict) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": ["nope"]})

    with pytest.raises(ApiError) as excinfo:
        make_client(handler).predict_batch([booking])
    assert str(excinfo.value).startswith("Dữ liệu không hợp lệ (422): ")


def test_health_maps_a_non_json_body(booking: dict) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>proxy</html>")

    with pytest.raises(ApiError) as excinfo:
        make_client(handler).health()
    assert str(excinfo.value) == "API trả về dữ liệu không phải JSON: <html>proxy</html>"


def test_predict_one_maps_a_non_json_body(booking: dict) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>proxy</html>")

    with pytest.raises(ApiError):
        make_client(handler).predict_one(booking)


def test_predict_batch_maps_a_non_json_body(booking: dict) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>proxy</html>")

    with pytest.raises(ApiError):
        make_client(handler).predict_batch([booking])


def test_predict_one_maps_a_redirect(booking: dict) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(307, headers={"location": "http://elsewhere/predict"}, text="moved")

    with pytest.raises(ApiError) as excinfo:
        make_client(handler).predict_one(booking)
    assert str(excinfo.value) == "API trả lỗi 307: moved"


def test_health_maps_a_redirect() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://elsewhere/health"}, text="moved")

    with pytest.raises(ApiError) as excinfo:
        make_client(handler).health()
    assert "302" in str(excinfo.value)


def test_predict_one_maps_503(booking: dict) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "No model loaded."})

    with pytest.raises(ApiError) as excinfo:
        make_client(handler).predict_one(booking)
    assert str(excinfo.value) == (
        "API chưa nạp được mô hình (503). Chạy `uv run invoke sample-model` rồi khởi động lại API."
    )


def test_predict_one_maps_other_status(booking: dict) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    with pytest.raises(ApiError) as excinfo:
        make_client(handler).predict_one(booking)
    assert str(excinfo.value) == "API trả lỗi 500: boom"


def test_health_rejects_a_body_that_is_not_the_health_schema() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"foo": 1})

    with pytest.raises(ApiError) as excinfo:
        make_client(handler).health()
    message = str(excinfo.value)
    assert message.startswith("API trả về dữ liệu không đúng định dạng: ")
    assert '{"foo"' in message


def test_predict_one_rejects_a_body_that_is_not_the_prediction_schema(booking: dict) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"foo": 1})

    with pytest.raises(ApiError) as excinfo:
        make_client(handler).predict_one(booking)
    message = str(excinfo.value)
    assert message.startswith("API trả về dữ liệu không đúng định dạng: ")
    assert '{"foo"' in message


def test_predict_batch_rejects_a_body_that_is_not_the_batch_schema(booking: dict) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"foo": 1})

    with pytest.raises(ApiError) as excinfo:
        make_client(handler).predict_batch([booking])
    message = str(excinfo.value)
    assert message.startswith("API trả về dữ liệu không đúng định dạng: ")
    assert '{"foo"' in message


def test_predict_one_rejects_a_probability_that_is_not_a_number(booking: dict) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "probability": "abc",
                "is_canceled": True,
                "threshold": 0.18,
                "model_version": "v1",
                "warnings": [],
            },
        )

    with pytest.raises(ApiError) as excinfo:
        make_client(handler).predict_one(booking)
    assert str(excinfo.value).startswith("API trả về dữ liệu không đúng định dạng: ")
