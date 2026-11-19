"""Request and response schemas of the prediction API (contract C1 in `docs/contracts.md`).

`Booking` holds the raw columns known at booking time, exactly `config.MODEL_INPUT_COLUMNS`. Pydantic rejects
malformed requests with 422; values that are well-formed but unseen in train (a new category, a value outside
the train range) are still predicted and reported as warnings, since such shifts are what drift monitoring
must see.
"""

from pydantic import BaseModel, ConfigDict, Field

MAX_BATCH_SIZE = 10_000


class Booking(BaseModel):
    """One booking as the booking system sends it, before any feature engineering."""

    model_config = ConfigDict(extra="forbid")

    hotel: str = Field(examples=["City Hotel"])
    arrival_date_month: str = Field(examples=["March"])
    meal: str = Field(examples=["BB"])
    market_segment: str = Field(examples=["Online TA"])
    distribution_channel: str = Field(examples=["TA/TO"])
    reserved_room_type: str = Field(examples=["A"])
    deposit_type: str = Field(examples=["No Deposit"])
    customer_type: str = Field(examples=["Transient"])
    lead_time: int = Field(ge=0, examples=[45])
    arrival_date_year: int = Field(ge=2000, le=2100, examples=[2017])
    arrival_date_week_number: int = Field(ge=1, le=53, examples=[11])
    arrival_date_day_of_month: int = Field(ge=1, le=31, examples=[14])
    stays_in_weekend_nights: int = Field(ge=0, examples=[1])
    stays_in_week_nights: int = Field(ge=0, examples=[2])
    adults: int = Field(ge=0, examples=[2])
    children: float | None = Field(default=None, ge=0, examples=[0])
    babies: int = Field(ge=0, examples=[0])
    is_repeated_guest: int = Field(ge=0, le=1, examples=[0])
    previous_cancellations: int = Field(ge=0, examples=[0])
    previous_bookings_not_canceled: int = Field(ge=0, examples=[0])
    agent: float | None = Field(default=None, examples=[9.0])
    company: float | None = Field(default=None, examples=[None])
    days_in_waiting_list: int = Field(ge=0, examples=[0])
    adr: float = Field(examples=[95.0])
    total_of_special_requests: int = Field(ge=0, examples=[1])
    rooms_in_booking: int | None = Field(
        default=None,
        ge=1,
        description="Rooms in the same booking. When omitted, it is counted from identical rows in the request.",
        examples=[1],
    )


class BatchRequest(BaseModel):
    """Several bookings scored in one call."""

    bookings: list[Booking] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class InputWarning(BaseModel):
    """A value the model accepts but did not see in train."""

    column: str
    issue: str = Field(examples=["unseen category"])
    rows: int


class Prediction(BaseModel):
    """Cancellation probability and decision for one booking."""

    probability: float = Field(ge=0, le=1, description="Probability that the booking is canceled.")
    is_canceled: bool = Field(description="probability >= threshold")


class PredictResponse(Prediction):
    """Response of `POST /predict`."""

    threshold: float
    model_version: str
    warnings: list[InputWarning] = []


class BatchResponse(BaseModel):
    """Response of `POST /predict/batch`, predictions in request order."""

    predictions: list[Prediction]
    threshold: float
    model_version: str
    warnings: list[InputWarning] = []


class HealthResponse(BaseModel):
    """Response of `GET /health`."""

    status: str = Field(examples=["ok"])
    model_loaded: bool
    model_version: str | None = None
