from pathlib import Path
from typing import Any

from streamlit.testing.v1 import AppTest

from hotel_booking.api_client import ApiClient, BatchResult, PredictResult, Prediction
from hotel_booking.config import MODEL_INPUT_COLUMNS
from hotel_booking.frontend import (
    ADVANCED_DEFAULTS,
    SOURCE_SAMPLE,
    SOURCE_UPLOAD,
    _next_batch_action,
    booking_payload,
    format_probability,
)

APP_PATH = Path(__file__).resolve().parents[1] / "src" / "hotel_booking" / "frontend.py"
SAMPLE_CSV_PATH = Path(__file__).resolve().parents[1] / "docs" / "examples" / "stream_sample.csv"
UPLOAD_NAME = "bookings.csv"
UPLOAD_BYTES = ("\n".join(SAMPLE_CSV_PATH.read_text(encoding="utf-8").splitlines()[:3]) + "\n").encode("utf-8")

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


def count_predict_batch(monkeypatch: Any) -> list[int]:
    """Replace `ApiClient.predict_batch` with a counter returning a fixed batch result.

    Args:
        monkeypatch: Pytest fixture used to patch the client class.

    Returns:
        The list the fake appends the payload size to on every call.
    """
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
    return calls


def batch_state(app: AppTest) -> dict[str, Any]:
    """Read the session state of the last run to assert on the batch keys.

    Args:
        app: Application under test, already run at least once.

    Returns:
        The filtered session state of the script run.
    """
    state: dict[str, Any] = app._tree.session_state.filtered_state
    return state


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
    state = batch_state(app)
    assert "batch_source" not in state
    assert "batch_upload_signature" not in state


def test_batch_results_are_scored_once_and_kept_across_reruns(monkeypatch) -> None:
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")
    calls = count_predict_batch(monkeypatch)
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


def test_next_batch_action_returns_to_the_upload_after_the_sample() -> None:
    assert _next_batch_action("file-a", "file-a", SOURCE_SAMPLE, use_sample=False, use_upload=True) == SOURCE_UPLOAD
    assert _next_batch_action("file-a", "file-a", SOURCE_SAMPLE, use_sample=False, use_upload=False) is None


def test_next_batch_action_finds_the_upload_after_a_clear() -> None:
    assert _next_batch_action("file-a", None, None, use_sample=False, use_upload=False) == SOURCE_UPLOAD


def test_next_batch_action_replaces_the_stored_file_and_prefers_the_sample() -> None:
    assert _next_batch_action("file-b", "file-a", SOURCE_UPLOAD, use_sample=False, use_upload=False) == SOURCE_UPLOAD
    assert _next_batch_action("file-a", "file-a", SOURCE_UPLOAD, use_sample=False, use_upload=False) is None
    assert _next_batch_action("file-a", "file-a", SOURCE_SAMPLE, use_sample=True, use_upload=True) == SOURCE_SAMPLE
    assert _next_batch_action(None, None, None, use_sample=True, use_upload=True) == SOURCE_SAMPLE
    assert _next_batch_action(None, None, None, use_sample=False, use_upload=True) is None


def test_use_upload_returns_to_the_uploaded_file_after_the_sample(monkeypatch) -> None:
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")
    calls = count_predict_batch(monkeypatch)
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    app.file_uploader[0].upload(UPLOAD_NAME, UPLOAD_BYTES, "text/csv")
    app.run()
    assert not app.exception
    assert calls == [2]
    assert any(f"dòng từ file {UPLOAD_NAME}" in element.value for element in app.caption)

    app.button(key="use_sample").click().run()
    assert not app.exception
    assert any("dòng từ file mẫu" in element.value for element in app.caption)

    app.button(key="use_upload").click().run()
    assert not app.exception
    assert any(f"dòng từ file {UPLOAD_NAME}" in element.value for element in app.caption)


def test_clear_reloads_the_upload_as_an_unscored_preview(monkeypatch) -> None:
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")
    calls = count_predict_batch(monkeypatch)
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    app.file_uploader[0].upload(UPLOAD_NAME, UPLOAD_BYTES, "text/csv")
    app.run()
    assert calls == [2]

    app.button(key="clear_batch").click().run()
    assert not app.exception
    assert calls == [2]
    assert len(app.dataframe) == 1
    assert any(f"Đã chọn 2 dòng từ file {UPLOAD_NAME}" in element.value for element in app.caption)

    app.button(key="score_batch").click().run()
    assert not app.exception
    assert calls == [2, 2]
    assert any("ngưỡng quyết định: 0.500" in element.value for element in app.caption)
