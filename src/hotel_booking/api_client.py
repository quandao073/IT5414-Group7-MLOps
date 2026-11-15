"""HTTP client for the hotel booking prediction API."""

from dataclasses import dataclass
from typing import Any, List

import httpx
import numpy as np
import pandas as pd

from hotel_booking.config import INPUT_COLUMNS
from hotel_booking.schemas import MAX_BATCH_SIZE


DEFAULT_API_URL: str = "http://localhost:8000"
DEFAULT_TIMEOUT: float = 10.0


@dataclass(frozen=True)
class Health:
    """Health check result."""

    status: str
    model_loaded: bool
    model_version: str | None


@dataclass(frozen=True)
class Prediction:
    """Prediction result for one booking."""

    probability: float
    is_canceled: bool


@dataclass(frozen=True)
class PredictResult:
    """Result of a single prediction call."""

    prediction: Prediction
    threshold: float
    model_version: str
    warnings: List[dict[str, Any]]


@dataclass(frozen=True)
class BatchResult:
    """Result of a batch prediction call."""

    predictions: List[Prediction]
    threshold: float
    model_version: str
    warnings: List[dict[str, Any]]


class ApiError(Exception):
    """User-facing error raised by the API client.

    Raised for transport failures such as a refused connection or a timeout, for HTTP error
    responses such as 503 or 422, and for request batches larger than `schemas.MAX_BATCH_SIZE`.
    `str(error)` is the Vietnamese message meant to be displayed to the user.
    """


def _convert_value(val: Any) -> Any:
    """Convert one cell to a JSON-serialisable Python value.

    Args:
        val: A cell value, possibly a NumPy scalar or a missing value.

    Returns:
        A plain Python value: `numpy.integer` becomes `int`, `numpy.floating` becomes
        `float`, and `NaN` or `pd.NA` becomes `None`.
    """
    if pd.isna(val):
        return None
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return float(val)
    return val


def csv_to_bookings(frame: pd.DataFrame) -> tuple[List[dict[str, Any]], List[str]]:
    """Convert a DataFrame of bookings into request payloads.

    Args:
        frame: Bookings with at least the `config.INPUT_COLUMNS` columns. A `rooms_in_booking`
            column is kept when present, otherwise the API infers it from identical rows.

    Returns:
        A tuple `(payloads, missing)`. `missing` lists the required columns absent from `frame`
        in `config.INPUT_COLUMNS` order; when it is non-empty the payload list is empty.
    """
    has_rooms_in_booking = "rooms_in_booking" in frame.columns
    rooms_in_booking_vals: list[int | None] = []
    if has_rooms_in_booking:
        for i in range(len(frame)):
            rooms_in_booking_vals.append(_convert_value(frame["rooms_in_booking"].iloc[i]))

    missing = [col for col in INPUT_COLUMNS if col not in frame.columns]
    if missing:
        return [], missing

    frame = frame.reindex(columns=INPUT_COLUMNS)

    records: List[dict[str, Any]] = []
    room_idx = 0
    for _, row in frame.iterrows():
        record: dict[str, Any] = {}
        for col in INPUT_COLUMNS:
            record[col] = _convert_value(row[col])
        if has_rooms_in_booking:
            record["rooms_in_booking"] = rooms_in_booking_vals[room_idx]
            room_idx += 1
        records.append(record)

    return records, []


class ApiClient:
    """HTTP client for the prediction API."""

    def __init__(
        self,
        base_url: str = DEFAULT_API_URL,
        timeout: float = DEFAULT_TIMEOUT,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Create a client backed by a single `httpx.Client`.

        Args:
            base_url: API root, by default `http://localhost:8000`.
            timeout: Seconds to wait for a response.
            transport: Optional httpx transport; tests pass an `httpx.MockTransport`.
        """
        self._client = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)
        self._base_url = base_url
        self._timeout = timeout

    @property
    def base_url(self) -> str:
        """Base URL of the API."""
        return self._base_url

    @property
    def timeout(self) -> float:
        """Timeout value used for requests."""
        return self._timeout

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        """Send one request and map every failure to an `ApiError`.

        Args:
            method: HTTP method, for example `GET` or `POST`.
            path: Path relative to `base_url`, for example `/predict`.
            **kwargs: Extra arguments forwarded to `httpx.Client.request`.

        Returns:
            The response when its status is below 400.

        Raises:
            ApiError: If the connection fails, the request times out, or the API answers with
                status 503, status 422, or any other status of 400 or more.
        """
        try:
            response = self._client.request(method, path, **kwargs)
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise ApiError(
                f"Không kết nối được API tại {self._base_url}. " f"Chạy `uv run invoke serve-api` để bật API."
            ) from e
        except httpx.TimeoutException as e:
            raise ApiError(f"API tại {self._base_url} không phản hồi sau {self._timeout:.0f} giây.") from e

        if response.status_code == 503:
            raise ApiError("API chưa nạp được mô hình (503). Chạy `uv run invoke sample-model` rồi khởi động lại API.")

        if response.status_code == 422:
            detail = response.json().get("detail", [])
            column_names = [item["loc"][1] for item in detail]
            raise ApiError("Dữ liệu không hợp lệ (422) ở cột: " + ", ".join(column_names))

        if response.status_code >= 400:
            raise ApiError(f"API trả lỗi {response.status_code}: {response.text[:200]}")

        return response

    def health(self) -> Health:
        """Report whether the API has a model loaded.

        Returns:
            The health payload with `status`, `model_loaded` and `model_version`.

        Raises:
            ApiError: If the request fails or the API returns an error status.
        """
        response = self._request("GET", "/health")
        data = response.json()
        return Health(
            status=data["status"],
            model_loaded=data["model_loaded"],
            model_version=data.get("model_version"),
        )

    def predict_one(self, booking: dict[str, Any]) -> PredictResult:
        """Score one booking.

        Args:
            booking: A payload whose keys are `config.MODEL_INPUT_COLUMNS`.

        Returns:
            The prediction with its probability, the decision threshold, the model version
            and any input warnings.

        Raises:
            ApiError: If the request fails or the API returns an error status.
        """
        response = self._request("POST", "/predict", json=booking)
        data = response.json()
        prediction = Prediction(probability=data["probability"], is_canceled=data["is_canceled"])
        return PredictResult(
            prediction=prediction,
            threshold=data["threshold"],
            model_version=data["model_version"],
            warnings=data.get("warnings", []),
        )

    def predict_batch(self, bookings: List[dict[str, Any]]) -> BatchResult:
        """Score several bookings in one request, in the order given.

        Args:
            bookings: Between 1 and `schemas.MAX_BATCH_SIZE` payloads whose keys are
                `config.MODEL_INPUT_COLUMNS`.

        Returns:
            The predictions in request order, with the decision threshold, the model version
            and any input warnings.

        Raises:
            ApiError: If there are more than `schemas.MAX_BATCH_SIZE` bookings, if the request
                fails, or if the API returns an error status.
        """
        if len(bookings) > MAX_BATCH_SIZE:
            raise ApiError("Chỉ chấm tối đa 10.000 bản ghi mỗi lần.")

        response = self._request("POST", "/predict/batch", json={"bookings": bookings})
        data = response.json()
        predictions = [
            Prediction(probability=p["probability"], is_canceled=p["is_canceled"]) for p in data["predictions"]
        ]
        return BatchResult(
            predictions=predictions,
            threshold=data["threshold"],
            model_version=data["model_version"],
            warnings=data.get("warnings", []),
        )
