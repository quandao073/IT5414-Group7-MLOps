from pathlib import Path
from typing import Any

from streamlit.testing.v1 import AppTest

from hotel_booking.api_client import ApiClient, BatchResult, PredictResult, Prediction
from hotel_booking.config import MODEL_INPUT_COLUMNS
from hotel_booking.frontend import ADVANCED_DEFAULTS, booking_payload, format_probability

APP_PATH = Path(__file__).resolve().parents[1] / "src" / "hotel_booking" / "frontend.py"

MAIN_VALUES: dict[str, Any] = {
    "hotel": "City Hotel",
    "arrival_date_year": 2017,
    "arrival_date_month": "March",
    "arrival_date_week_number": 11,
    "arrival_date_day_of_month": 14,
    "lead_time": 45,
    "stays_in_weekend_nights": 1,
    "stays_in_week_nights": 2,
    "adults": 2,
    "children": 0,
    "deposit_type": "No Deposit",
    "customer_type": "Transient",
    "adr": 95.0,
}


def test_booking_payload_merges_advanced_defaults() -> None:
    payload = booking_payload(MAIN_VALUES)
    assert set(payload) == set(MODEL_INPUT_COLUMNS)
    assert payload["adults"] == 2
    assert payload["agent"] is None
    assert payload["company"] is None
    assert payload["rooms_in_booking"] is None
    assert payload["meal"] == ADVANCED_DEFAULTS["meal"]


def test_format_probability() -> None:
    assert format_probability(0.2773) == "27.7%"


def test_result_panel_shows_probability_and_threshold(monkeypatch) -> None:
    import streamlit as st

    captured: dict[str, Any] = {}

    def fake_metric(label: str, value: Any, **kwargs: Any) -> None:
        captured.setdefault("metrics", []).append((label, value))

    def fake_caption(text: str) -> None:
        captured["caption"] = text

    monkeypatch.setattr(st, "metric", fake_metric)
    monkeypatch.setattr(st, "caption", fake_caption)
    monkeypatch.setattr(st, "progress", lambda *args, **kwargs: None)
    monkeypatch.setattr(st, "warning", lambda *args, **kwargs: None)

    from hotel_booking.frontend import result_panel

    result_panel(
        PredictResult(
            prediction=Prediction(probability=0.5, is_canceled=True),
            threshold=0.1825,
            model_version="v1",
            warnings=[],
        )
    )
    assert captured["metrics"] == [("Xác suất hủy", "50.0%")]
    assert "0.183" in captured["caption"]
    assert "v1" in captured["caption"]


def test_frontend_renders_and_reports_api_down(monkeypatch) -> None:
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    assert not app.exception
    assert app.title[0].value == "Dự báo hủy phòng khách sạn"
    assert app.sidebar.text_input[0].value == "http://127.0.0.1:9"
    assert any("Không kết nối được API" in element.value for element in app.error)


def test_batch_preview_survives_an_unrelated_rerun(monkeypatch) -> None:
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    app.button(key="use_sample").click().run()
    assert not app.exception
    assert len(app.dataframe) == 1
    assert any("dòng từ file mẫu" in element.value for element in app.caption)

    app.sidebar.text_input[0].set_value("http://127.0.0.1:8").run()
    assert not app.exception
    assert len(app.dataframe) == 1
    assert any("dòng từ file mẫu" in element.value for element in app.caption)


def test_clear_batch_drops_the_stored_preview(monkeypatch) -> None:
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    app.button(key="use_sample").click().run()
    assert len(app.dataframe) == 1

    app.button(key="clear_batch").click().run()
    assert not app.exception
    assert len(app.dataframe) == 0
    assert any("Chọn file CSV hoặc dùng file mẫu" in element.value for element in app.caption)


def test_batch_results_are_scored_once_and_kept_across_reruns(monkeypatch) -> None:
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")
    calls: list[int] = []

    def fake_predict_batch(self: ApiClient, payloads: list[dict[str, Any]]) -> BatchResult:
        calls.append(len(payloads))
        return BatchResult(
            predictions=[Prediction(probability=0.4, is_canceled=True) for _ in payloads],
            threshold=0.5,
            model_version="v1",
            warnings=[],
        )

    monkeypatch.setattr(ApiClient, "predict_batch", fake_predict_batch)
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    app.button(key="use_sample").click().run()
    assert not app.exception
    assert calls == [20]
    assert len(app.dataframe) == 1
    assert any("ngưỡng quyết định: 0.500" in element.value for element in app.caption)

    app.sidebar.text_input[0].set_value("http://127.0.0.1:8").run()
    assert not app.exception
    assert calls == [20]
    assert len(app.dataframe) == 1
    assert any("ngưỡng quyết định: 0.500" in element.value for element in app.caption)

    app.button(key="score_batch").click().run()
    assert not app.exception
    assert calls == [20, 20]
    assert len(app.dataframe) == 1
