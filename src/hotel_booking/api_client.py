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
    """Error raised when the API returns an error."""

    pass


def _convert_value(val: Any) -> Any:
    """Convert a value to a JSON-serializable Python type.

    Parameters
    ----------
    val: Any
        Value potentially containing numpy types.

    Returns
    -------
        Converted value with numpy integers/floats converted to Python types,
        and NaN/pd.NA converted to None.
    """
    if pd.isna(val):
        return None
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return float(val)
    if val is None:
        return None
    return val


def csv_to_bookings(frame: pd.DataFrame) -> tuple[List[dict[str, Any]], List[str]]:
    """Convert a DataFrame of bookings into a list of payloads.

    Parameters
    ----------
    frame: pd.DataFrame
        DataFrame containing booking data.

    Returns
    -------
        Tuple of (payloads, missing_columns). If any INPUT_COLUMNS are missing
        from the frame, payloads is an empty list and missing lists the column names.
    """
    # Save rooms_in_booking values before reindex removes the column
    has_rooms_in_booking = "rooms_in_booking" in frame.columns
    rooms_in_booking_vals: list[int | None] = []
    if has_rooms_in_booking:
        for i in range(len(frame)):
            v = frame["rooms_in_booking"].iloc[i]
            rooms_in_booking_vals.append(_convert_value(v))

    missing = [col for col in INPUT_COLUMNS if col not in frame.columns]

    if missing:
        return [], missing

    # All INPUT_COLUMNS are present; reindex to ensure column order
    frame = frame.reindex(columns=INPUT_COLUMNS)

    records: List[dict[str, Any]] = []
    room_idx = 0
    for _, row in frame.iterrows():
        record: dict[str, Any] = {}
        for col in INPUT_COLUMNS:
            val = row[col]
            record[col] = _convert_value(val)
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
        """Make an HTTP request and handle connection errors, timeouts, and HTTP errors.

        Parameters
        ----------
        method: str
            HTTP method (GET, POST, etc.).
        path: str
            API path relative to base_url.
        **kwargs: Any
            Additional arguments to pass to httpx.Client.request.

        Returns
        -------
            httpx.Response on success.

        Raises
        ------
        ApiError
            If a connection error, timeout, or HTTP error occurs.
        """
        try:
            response = self._client.request(method, path, **kwargs)
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise ApiError(
                f"Không kết nối được API tới {self._base_url}. " f"Chạy `uv run invoke serve-api` để bắt API."
            ) from e
        except httpx.TimeoutException as e:
            raise ApiError(f"API tới {self._base_url} không phản hồi sau {self._timeout:.0f} giây.") from e

        # Map HTTP status codes to user-facing Vietnamese error messages
        if response.status_code == 503:
            raise ApiError(
                "API chưa nạp được mô hình (503). " "Chạy `uv run invoke sample-model` để khởi động lại API."
            )

        if response.status_code == 422:
            detail = response.json().get("detail", [])
            column_names = [item["loc"][1] for item in detail]
            raise ApiError("Dữ liệu không hợp lệ (422) " + ", ".join(column_names))

        if response.status_code >= 400:
            raise ApiError(f"API trả về {response.status_code}: {response.text[:200]}")

        return response

    def health(self) -> Health:
        """Check API health.

        Returns
        -------
            Health dataclass with status, model_loaded, and model_version.

        Raises
        ------
        ApiError
            If the API returns a non-200 status code.
        """
        response = self._request("GET", "/health")

        if response.status_code == 200:
            data = response.json()
            return Health(
                status=data["status"],
                model_loaded=data["model_loaded"],
                model_version=data.get("model_version"),
            )

        # Status code already raised by _request, but fallback for safety
        raise ApiError(f"API trả về {response.status_code}: {response.text[:200]}")

    def predict_one(self, booking: dict[str, Any]) -> PredictResult:
        """Make a single prediction.

        Parameters
        ----------
        booking: dict[str, Any]
            A booking payload with keys matching MODEL_INPUT_COLUMNS.

        Returns
        -------
            PredictResult with prediction, threshold, model_version, and warnings.

        Raises
        ------
        ApiError
            If the API returns a non-200 status code or a connection error occurs.
        """
        response = self._request("POST", "/predict", json=booking)

        if response.status_code == 200:
            data = response.json()
            prediction = Prediction(
                probability=data["probability"],
                is_canceled=data["is_canceled"],
            )
            return PredictResult(
                prediction=prediction,
                threshold=data["threshold"],
                model_version=data["model_version"],
                warnings=data.get("warnings", []),
            )

        # Status code already raised by _request, but fallback for safety
        raise ApiError(f"API trả về {response.status_code}: {response.text[:200]}")

    def predict_batch(self, bookings: List[dict[str, Any]]) -> BatchResult:
        """Make batch predictions.

        Parameters
        ----------
        bookings: list[dict[str, Any]]
            List of booking payloads.

        Returns
        -------
            BatchResult with predictions, threshold, model_version, and warnings.

        Raises
        ------
        ApiError
            If the number of bookings exceeds MAX_BATCH_SIZE or the API returns
            a non-200 status code, or a connection error occurs.
        """
        if len(bookings) > MAX_BATCH_SIZE:
            raise ApiError("Chỉ chấm tối đa 10.000 bản ghi mỗi lần.")

        response = self._request("POST", "/predict/batch", json={"bookings": bookings})

        if response.status_code == 200:
            data = response.json()
            predictions = [
                Prediction(
                    probability=p["probability"],
                    is_canceled=p["is_canceled"],
                )
                for p in data["predictions"]
            ]
            return BatchResult(
                predictions=predictions,
                threshold=data["threshold"],
                model_version=data["model_version"],
                warnings=data.get("warnings", []),
            )

        # Status code already raised by _request, but fallback for safety
        raise ApiError(f"API trả về {response.status_code}: {response.text[:200]}")
