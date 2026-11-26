"""Shared constants: target, temporal split boundaries and the input data contract."""

RANDOM_STATE = 42

TARGET = "is_canceled"
ID_COLUMN = "booking_id"
TIME_COLUMN = "arrival_date"
RAW_FILE = "hotel_bookings.csv"

# Split by arrival date: train up to TRAIN_END, test up to TEST_END, the rest is the production holdout.
TRAIN_END = "2016-12-31"
TEST_END = "2017-04-30"
SPLITS = ("train", "test", "production_holdout")

MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

# Direct leakage: these columns encode the outcome and are removed from every output file.
LEAKAGE_COLUMNS = ["reservation_status", "reservation_status_date"]

# Known only after the booking is made: kept for analysis, never used as model inputs.
POST_BOOKING_COLUMNS = ["assigned_room_type", "booking_changes", "required_car_parking_spaces", "country"]

# Number of rows sharing the same booking-time details, i.e. rooms in the same booking.
GROUP_SIZE_COLUMN = "rooms_in_booking"

# Raw columns known at booking time: the inputs of the model pipeline.
RAW_CATEGORICAL_INPUTS = [
    "hotel",
    "arrival_date_month",
    "meal",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
]
RAW_NUMERIC_INPUTS = [
    "lead_time",
    "arrival_date_year",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "agent",
    "company",
    "days_in_waiting_list",
    "adr",
    "total_of_special_requests",
]
INPUT_COLUMNS = RAW_CATEGORICAL_INPUTS + RAW_NUMERIC_INPUTS
MODEL_INPUT_COLUMNS = INPUT_COLUMNS + [GROUP_SIZE_COLUMN]
