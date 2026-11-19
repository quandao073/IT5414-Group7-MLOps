from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from hotel_booking import config, data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / config.RAW_FILE

CATEGORIES = {
    "hotel": ["City Hotel", "Resort Hotel"],
    "meal": ["BB", "HB", "SC", "Undefined"],
    "market_segment": ["Online TA", "Offline TA/TO", "Groups", "Direct", "Corporate"],
    "distribution_channel": ["TA/TO", "Direct", "Corporate"],
    "reserved_room_type": ["A", "D", "E"],
    "deposit_type": ["No Deposit", "Non Refund"],
    "customer_type": ["Transient", "Transient-Party", "Contract"],
}


def make_raw_bookings(n: int = 800, seed: int = 0) -> pd.DataFrame:
    """Synthetic bookings in the raw Kaggle format, spanning every split, with both classes."""
    rng = np.random.default_rng(seed)
    dates = pd.Series(rng.choice(pd.date_range("2015-07-01", "2017-08-31"), n))
    frame = pd.DataFrame({column: rng.choice(values, n) for column, values in CATEGORIES.items()})
    frame["lead_time"] = rng.integers(0, 400, n)
    frame["arrival_date_year"] = dates.dt.year
    frame["arrival_date_month"] = dates.dt.month_name()
    frame["arrival_date_week_number"] = dates.dt.isocalendar().week.astype(int).to_numpy()
    frame["arrival_date_day_of_month"] = dates.dt.day
    frame["stays_in_weekend_nights"] = rng.integers(0, 3, n)
    frame["stays_in_week_nights"] = rng.integers(1, 6, n)
    frame["adults"] = rng.integers(1, 4, n)
    frame["children"] = rng.choice([0.0, 1.0, np.nan], n, p=[0.85, 0.1, 0.05])
    frame["babies"] = rng.choice([0, 1], n, p=[0.95, 0.05])
    frame["is_repeated_guest"] = rng.choice([0, 1], n, p=[0.9, 0.1])
    frame["previous_cancellations"] = rng.choice([0, 1], n, p=[0.9, 0.1])
    frame["previous_bookings_not_canceled"] = rng.choice([0, 2], n, p=[0.9, 0.1])
    frame["agent"] = rng.choice([9.0, 240.0, np.nan], n)
    frame["company"] = rng.choice([np.nan, 40.0], n, p=[0.9, 0.1])
    frame["days_in_waiting_list"] = rng.choice([0, 10], n, p=[0.95, 0.05])
    frame["adr"] = rng.uniform(40, 250, n).round(2)
    frame["total_of_special_requests"] = rng.integers(0, 3, n)
    frame["assigned_room_type"] = frame["reserved_room_type"]
    frame["booking_changes"] = rng.integers(0, 2, n)
    frame["required_car_parking_spaces"] = rng.choice([0, 1], n, p=[0.9, 0.1])
    frame["country"] = rng.choice(["PRT", "GBR", "FRA"], n)
    signal = frame["lead_time"] / 400 + (frame["deposit_type"] == "Non Refund") - frame["total_of_special_requests"] / 4
    frame[config.TARGET] = (signal + rng.normal(0, 0.3, n) > 0.4).astype(int)
    frame["reservation_status"] = np.where(frame[config.TARGET] == 1, "Canceled", "Check-Out")
    frame["reservation_status_date"] = dates.dt.strftime("%Y-%m-%d")
    frame.loc[0, "adults"] = 0
    frame.loc[0, ["children", "babies"]] = 0
    return frame


@pytest.fixture(scope="session")
def raw_bookings() -> pd.DataFrame:
    """Synthetic raw bookings, including one row with no guests that cleaning must drop."""
    return make_raw_bookings()


@pytest.fixture(scope="session")
def sample(raw_bookings: pd.DataFrame) -> pd.DataFrame:
    """Cleaned synthetic bookings, small enough to fit models quickly."""
    frame, _ = data.clean_bookings(raw_bookings)
    return frame


@pytest.fixture(scope="session")
def real_raw() -> pd.DataFrame:
    """The real Kaggle dataset; tests using it are skipped when it has not been pulled with DVC."""
    if not RAW_PATH.exists():
        pytest.skip(f"{RAW_PATH} not found; run `dvc pull`.")
    return data.load_raw(RAW_PATH)


@pytest.fixture
def booking() -> dict[str, Any]:
    """One raw booking as the booking system would send it."""
    return {
        "hotel": "City Hotel",
        "arrival_date_month": "March",
        "meal": "BB",
        "market_segment": "Online TA",
        "distribution_channel": "TA/TO",
        "reserved_room_type": "A",
        "deposit_type": "No Deposit",
        "customer_type": "Transient",
        "lead_time": 45,
        "arrival_date_year": 2017,
        "arrival_date_week_number": 11,
        "arrival_date_day_of_month": 14,
        "stays_in_weekend_nights": 1,
        "stays_in_week_nights": 2,
        "adults": 2,
        "children": 0,
        "babies": 0,
        "is_repeated_guest": 0,
        "previous_cancellations": 0,
        "previous_bookings_not_canceled": 0,
        "agent": 9.0,
        "company": None,
        "days_in_waiting_list": 0,
        "adr": 95.0,
        "total_of_special_requests": 1,
        "rooms_in_booking": 1,
    }
