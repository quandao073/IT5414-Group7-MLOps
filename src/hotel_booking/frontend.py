"""Streamlit frontend for the hotel booking cancellation API (contracts C1 and C5)."""

import os
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from hotel_booking.api_client import (
    DEFAULT_API_URL,
    ApiClient,
    ApiError,
    BatchResult,
    PredictResult,
    csv_to_bookings,
)
from hotel_booking.config import MODEL_INPUT_COLUMNS, MONTHS
from hotel_booking.schemas import MAX_BATCH_SIZE

st.set_page_config(page_title="Hotel booking cancellation", layout="wide")
st.title("Dự báo hủy phòng khách sạn")

SAMPLE_CSV = Path(__file__).resolve().parents[2] / "docs" / "examples" / "stream_sample.csv"
THRESHOLD_PRECISION = Decimal("0.001")

CATEGORY_CHOICES: dict[str, list[str]] = {
    "hotel": ["City Hotel", "Resort Hotel"],
    "arrival_date_month": MONTHS,
    "meal": ["BB", "HB", "SC", "Undefined"],
    "market_segment": ["Online TA", "Offline TA/TO", "Groups", "Direct", "Corporate"],
    "distribution_channel": ["TA/TO", "Direct", "Corporate"],
    "reserved_room_type": ["A", "D", "E"],
    "deposit_type": ["No Deposit", "Non Refund"],
    "customer_type": ["Transient", "Transient-Party", "Contract"],
}

ADVANCED_DEFAULTS: dict[str, Any] = {
    "meal": "BB",
    "market_segment": "Online TA",
    "distribution_channel": "TA/TO",
    "reserved_room_type": "A",
    "babies": 0,
    "is_repeated_guest": 0,
    "previous_cancellations": 0,
    "previous_bookings_not_canceled": 0,
    "agent": None,
    "company": None,
    "days_in_waiting_list": 0,
    "total_of_special_requests": 0,
    "rooms_in_booking": None,
}

FIELD_LABELS: dict[str, str] = {
    "hotel": "Loại khách sạn",
    "arrival_date_year": "Năm đến",
    "arrival_date_month": "Tháng đến",
    "arrival_date_week_number": "Tuần trong năm",
    "arrival_date_day_of_month": "Ngày trong tháng",
    "lead_time": "Số ngày đặt trước",
    "stays_in_weekend_nights": "Đêm cuối tuần",
    "stays_in_week_nights": "Đêm ngày thường",
    "adults": "Người lớn",
    "children": "Trẻ em",
    "deposit_type": "Loại đặt cọc",
    "customer_type": "Loại khách",
    "adr": "Giá mỗi đêm (ADR)",
    "meal": "Gói bữa ăn",
    "market_segment": "Phân khúc thị trường",
    "distribution_channel": "Kênh phân phối",
    "reserved_room_type": "Loại phòng đặt",
    "babies": "Em bé",
    "is_repeated_guest": "Khách quay lại (0/1)",
    "previous_cancellations": "Số lần hủy trước đó",
    "previous_bookings_not_canceled": "Số booking trước không hủy",
    "agent": "Mã đại lý",
    "company": "Mã công ty",
    "days_in_waiting_list": "Ngày trong danh sách chờ",
    "total_of_special_requests": "Số yêu cầu đặc biệt",
}

MAIN_DEFAULTS: dict[str, Any] = {
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

MAIN_FIELDS = [
    "hotel",
    "arrival_date_year",
    "arrival_date_month",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "deposit_type",
    "customer_type",
    "adr",
]

ADVANCED_FIELDS = [
    "meal",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "agent",
    "company",
    "days_in_waiting_list",
    "total_of_special_requests",
]

NUMERIC_SPECS: dict[str, tuple[float, float | None, float]] = {
    "arrival_date_year": (2000, 2100, 1),
    "arrival_date_week_number": (1, 53, 1),
    "arrival_date_day_of_month": (1, 31, 1),
    "lead_time": (0, None, 1),
    "stays_in_weekend_nights": (0, None, 1),
    "stays_in_week_nights": (0, None, 1),
    "adults": (0, None, 1),
    "children": (0, None, 1),
    "adr": (0.0, None, 0.01),
    "babies": (0, None, 1),
    "is_repeated_guest": (0, 1, 1),
    "previous_cancellations": (0, None, 1),
    "previous_bookings_not_canceled": (0, None, 1),
    "agent": (0, None, 1),
    "company": (0, None, 1),
    "days_in_waiting_list": (0, None, 1),
    "total_of_special_requests": (0, None, 1),
}


def api_url_from_env() -> str:
    """Read the API root from the environment, falling back to the local default.

    Returns:
        The value of `API_URL` without a trailing slash, or `http://localhost:8000` when the
        variable is not set.
    """
    load_dotenv()
    return os.getenv("API_URL", DEFAULT_API_URL).rstrip("/")


def booking_payload(values: dict[str, Any]) -> dict[str, Any]:
    """Build one request payload from the form values.

    Args:
        values: Values collected from the form, with the main fields and any advanced field the
            user edited.

    Returns:
        A payload holding exactly the `config.MODEL_INPUT_COLUMNS` keys: `values` overriding
        `ADVANCED_DEFAULTS` for the advanced fields the user did not touch.
    """
    merged = {**ADVANCED_DEFAULTS, **values}
    return {column: merged[column] for column in MODEL_INPUT_COLUMNS}


def format_probability(probability: float) -> str:
    """Format a cancellation probability as a Vietnamese percentage.

    Args:
        probability: Probability between 0 and 1.

    Returns:
        The probability with one decimal, for example `27.7%`.
    """
    return f"{probability:.1%}"


def format_threshold(threshold: float) -> str:
    """Format the decision threshold with three decimals, rounding halves up.

    Args:
        threshold: Decision threshold between 0 and 1.

    Returns:
        The threshold with three decimals, for example `0.183` for `0.1825`. A plain `:.3f`
        would print `0.182`, because the binary value of `0.1825` is slightly below the decimal
        one.
    """
    return str(Decimal(repr(threshold)).quantize(THRESHOLD_PRECISION, rounding=ROUND_HALF_UP))


def render_warnings(warnings: list[dict[str, Any]]) -> None:
    """Show one warning per input the model accepts but did not see in train.

    Args:
        warnings: Input warnings as returned by the API, each with `column`, `issue` and `rows`.
    """
    for warning in warnings:
        st.warning(f"{warning['column']}: {warning['issue']} ({warning['rows']} dòng)")


def result_panel(result: PredictResult) -> None:
    """Render the probability, the decision and any input warnings of one prediction.

    Args:
        result: The prediction returned by `ApiClient.predict_one`.
    """
    st.metric("Xác suất hủy", format_probability(result.prediction.probability))
    st.progress(result.prediction.probability)
    st.caption(
        f"Ngưỡng quyết định: {format_threshold(result.threshold)} · is_canceled = {result.prediction.is_canceled}"
        f" · model {result.model_version}"
    )
    render_warnings(result.warnings)


def _render_widget(field: str, default: Any) -> Any:
    """Render the input widget of one booking field and return the chosen value.

    Args:
        field: Column name of `config.MODEL_INPUT_COLUMNS`, which selects the widget kind.
        default: Initial value of the widget.

    Returns:
        The value the user selected, or `None` when a number input is left empty.
    """
    if field in CATEGORY_CHOICES:
        choices = CATEGORY_CHOICES[field]
        return st.selectbox(FIELD_LABELS[field], choices, index=choices.index(default))
    minimum, maximum, step = NUMERIC_SPECS[field]
    return st.number_input(FIELD_LABELS[field], min_value=minimum, max_value=maximum, value=default, step=step)


def render_sidebar(client: ApiClient) -> None:
    """Render the API URL input and the health check of the API.

    Args:
        client: Client whose `base_url` pre-fills the URL input and whose health is shown.
    """
    with st.sidebar:
        st.text_input("API URL", value=client.base_url)
        try:
            health = client.health()
        except ApiError as exc:
            st.error(str(exc))
        else:
            st.success(f"API sẵn sàng · model {health.model_version or 'chưa nạp'}")


def render_single_tab(client: ApiClient) -> None:
    """Render the single booking tab: the main fields, the advanced expander and the verdict.

    Args:
        client: Client used to score the booking when the user submits the form.
    """
    values: dict[str, Any] = {}
    with st.form("single_booking"):
        columns = st.columns(3)
        for index, field in enumerate(MAIN_FIELDS):
            with columns[index % len(columns)]:
                values[field] = _render_widget(field, MAIN_DEFAULTS[field])
        with st.expander("Thông tin nâng cao"):
            advanced_columns = st.columns(3)
            for index, field in enumerate(ADVANCED_FIELDS):
                with advanced_columns[index % len(advanced_columns)]:
                    values[field] = _render_widget(field, ADVANCED_DEFAULTS[field])
        submitted = st.form_submit_button("Dự báo")

    if not submitted:
        return

    try:
        result = client.predict_one(booking_payload(values))
    except ApiError as exc:
        st.error(str(exc))
        return

    result_panel(result)


def _read_batch_frame(uploaded: Any) -> pd.DataFrame | None:
    """Read the uploaded CSV, or the sample file when the user asked for it.

    Args:
        uploaded: File object returned by `st.file_uploader`, or `None`.

    Returns:
        The uploaded bookings, or `None` when the user has not chosen a file yet.
    """
    if st.button("Dùng file mẫu"):
        return pd.read_csv(SAMPLE_CSV)
    if uploaded is None:
        return None
    return pd.read_csv(uploaded)


def render_batch_tab(client: ApiClient) -> None:
    """Render the CSV tab: upload the file, score it and offer the results as a download.

    Args:
        client: Client used to score the batch when the user uploads a file.
    """
    uploaded = st.file_uploader("Chọn file CSV", type="csv")
    frame = _read_batch_frame(uploaded)
    if frame is None:
        return

    payloads, missing = csv_to_bookings(frame)
    if missing:
        st.error("File thiếu cột: " + ", ".join(missing))
        return
    if not payloads:
        st.error("File không có dòng dữ liệu nào.")
        return
    if len(payloads) > MAX_BATCH_SIZE:
        st.error(f"Chỉ chấm tối đa {MAX_BATCH_SIZE} dòng mỗi lần; file có {len(payloads)} dòng.")
        return

    try:
        result = client.predict_batch(payloads)
    except ApiError as exc:
        st.error(str(exc))
        return

    output = _results_frame(frame, result)
    st.dataframe(output)
    st.metric("Tỉ lệ hủy dự đoán", format_probability(float(output["probability"].mean())))
    st.caption(f"Ngưỡng quyết định: {format_threshold(result.threshold)} · model {result.model_version}")
    render_warnings(result.warnings)
    st.download_button(
        "Tải kết quả CSV",
        data=output.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv",
    )


def _results_frame(frame: pd.DataFrame, result: BatchResult) -> pd.DataFrame:
    """Append the probability and the decision of every row to the uploaded bookings.

    Args:
        frame: Bookings as read from the CSV, in request order.
        result: Predictions returned by `ApiClient.predict_batch`, in the same order.

    Returns:
        A copy of `frame` with the `probability` and `is_canceled` columns added.
    """
    output = frame.copy()
    output["probability"] = [prediction.probability for prediction in result.predictions]
    output["is_canceled"] = [prediction.is_canceled for prediction in result.predictions]
    return output


def main() -> None:
    """Render the sidebar and the two prediction tabs."""
    client = ApiClient(api_url_from_env())
    render_sidebar(client)
    single_tab, batch_tab = st.tabs(["Một booking", "Upload CSV"])
    with single_tab:
        render_single_tab(client)
    with batch_tab:
        render_batch_tab(client)


if __name__ == "__main__":
    main()
