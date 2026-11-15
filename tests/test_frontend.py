import csv
import io
from pathlib import Path
from typing import Any

import pandas as pd
from streamlit.testing.v1 import AppTest

from hotel_booking import api_client, frontend
from hotel_booking.api_client import ApiClient, ApiError, BatchResult, PredictResult, Prediction
from hotel_booking.config import MODEL_INPUT_COLUMNS
from hotel_booking.frontend import (
    ADVANCED_DEFAULTS,
    SOURCE_SAMPLE,
    SOURCE_UPLOAD,
    _next_batch_action,
    booking_payload,
    format_probability,
)
from hotel_booking.schemas import MAX_BATCH_SIZE

APP_PATH = Path(__file__).resolve().parents[1] / "src" / "hotel_booking" / "frontend.py"
SAMPLE_CSV_PATH = Path(__file__).resolve().parents[1] / "docs" / "examples" / "stream_sample.csv"
SAMPLE_LINES = SAMPLE_CSV_PATH.read_text(encoding="utf-8").splitlines()
UPLOAD_NAME = "bookings.csv"
UPLOAD_CSV_TEXT = "\n".join(SAMPLE_LINES[:3]) + "\n"
UPLOAD_BYTES = UPLOAD_CSV_TEXT.encode("utf-8")


def csv_without_column(text: str, column: str) -> str:
    """Remove one column from a CSV document.

    Args:
        text: CSV document, header first.
        column: Name of the column to drop from every row.

    Returns:
        The document without that column, terminated by a newline.
    """
    rows = list(csv.reader(io.StringIO(text)))
    dropped = rows[0].index(column)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    for row in rows:
        writer.writerow([cell for position, cell in enumerate(row) if position != dropped])
    return buffer.getvalue()


MISSING_COLUMN_BYTES = csv_without_column(UPLOAD_CSV_TEXT, "lead_time").encode("utf-8")
OVERSIZED_BYTES = (SAMPLE_LINES[0] + "\n" + (SAMPLE_LINES[1] + "\n") * (MAX_BATCH_SIZE + 1)).encode("utf-8")

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


def count_read_csv(monkeypatch: Any) -> list[str]:
    """Count the calls the frontend makes to `pandas.read_csv`.

    Args:
        monkeypatch: Pytest fixture used to patch `pandas.read_csv`.

    Returns:
        The list the delegate appends the first argument of every call to.
    """
    calls: list[str] = []
    real_read_csv = pd.read_csv

    def fake_read_csv(*args: Any, **kwargs: Any) -> pd.DataFrame:
        calls.append(str(args[0]) if args else "")
        frame: pd.DataFrame = real_read_csv(*args, **kwargs)
        return frame

    monkeypatch.setattr(frontend.pd, "read_csv", fake_read_csv)
    return calls


def spy_csv_to_bookings(monkeypatch: Any) -> list[int]:
    """Record the frames passed to `csv_to_bookings` while still converting them.

    The frontend script binds the function with `from hotel_booking.api_client import ...` on every
    run, so the patch has to live on the `api_client` module.

    Args:
        monkeypatch: Pytest fixture used to patch the conversion function.

    Returns:
        The list the spy appends the row count of every frame to.
    """
    calls: list[int] = []
    real_convert = api_client.csv_to_bookings

    def spy(frame: pd.DataFrame) -> tuple[list[dict[str, Any]], list[str]]:
        calls.append(len(frame))
        return real_convert(frame)

    monkeypatch.setattr(api_client, "csv_to_bookings", spy)
    return calls


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
    assert app.sidebar.text_input[0].proto.disabled
    assert any("Không kết nối được API" in element.value for element in app.error)


def test_batch_preview_survives_an_unrelated_rerun(monkeypatch) -> None:
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    app.button(key="use_sample").click().run()
    assert not app.exception
    assert len(app.dataframe) == 1
    assert any("dòng từ file mẫu" in element.value for element in app.caption)

    app.run()
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

    app.run()
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
    assert "use_upload" not in [button.key for button in app.button]

    app.button(key="use_sample").click().run()
    assert not app.exception
    assert any("dòng từ file mẫu" in element.value for element in app.caption)
    assert "use_upload" in [button.key for button in app.button]

    app.button(key="use_upload").click().run()
    assert not app.exception
    assert any(f"dòng từ file {UPLOAD_NAME}" in element.value for element in app.caption)

    app.run()
    assert not app.exception
    assert any(f"dòng từ file {UPLOAD_NAME}" in element.value for element in app.caption)
    assert "use_upload" not in [button.key for button in app.button]


def test_rejected_upload_is_not_read_again_on_the_next_rerun(monkeypatch) -> None:
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")
    reads = count_read_csv(monkeypatch)
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    app.file_uploader[0].upload("missing_column.csv", MISSING_COLUMN_BYTES, "text/csv")
    app.run()
    assert not app.exception
    assert any("File thiếu cột: lead_time" in element.value for element in app.error)
    assert len(reads) == 1

    app.run()
    assert not app.exception
    assert any("File thiếu cột: lead_time" in element.value for element in app.error)
    assert len(reads) == 1
    assert not any("Chọn file CSV hoặc dùng file mẫu" in element.value for element in app.caption)


def test_oversized_upload_is_rejected_before_conversion(monkeypatch) -> None:
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")
    conversions = spy_csv_to_bookings(monkeypatch)
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    app.file_uploader[0].upload("too_big.csv", OVERSIZED_BYTES, "text/csv")
    app.run()
    assert not app.exception
    assert any(f"Chỉ chấm tối đa {MAX_BATCH_SIZE} dòng mỗi lần" in element.value for element in app.error)
    assert conversions == []


def test_batch_error_disappears_when_a_later_attempt_succeeds(monkeypatch) -> None:
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")
    attempts: list[int] = []

    def flaky_predict_batch(self: ApiClient, payloads: list[dict[str, Any]]) -> BatchResult:
        attempts.append(len(payloads))
        if len(attempts) == 1:
            raise ApiError("Batch lỗi thử nghiệm.")
        return BatchResult(
            predictions=[Prediction(probability=0.4, is_canceled=False) for _ in payloads],
            threshold=0.5,
            model_version="v1",
            warnings=[],
        )

    monkeypatch.setattr(ApiClient, "predict_batch", flaky_predict_batch)
    app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()
    app.button(key="use_sample").click().run()
    assert not app.exception
    assert attempts == [20]
    assert any("Batch lỗi thử nghiệm." in element.value for element in app.error)
    assert len(app.metric) == 0

    app.button(key="score_batch").click().run()
    assert not app.exception
    assert attempts == [20, 20]
    assert not any("Batch lỗi thử nghiệm." in element.value for element in app.error)
    assert app.metric[0].value == "40.0%"


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
