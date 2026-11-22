"""Strict checks on the processed splits with pandera: columns, types, value ranges, nulls and time windows.

`data.validate_inputs` stays the soft check the API runs on single bookings (it only produces warnings); this module
is the hard gate on whole splits: the DVC `validate` stage runs it on `data/processed` and stops the pipeline before
training or drift reports when a split fails.
"""

import json
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
import typer

from hotel_booking.config import (
    GROUP_SIZE_COLUMN,
    ID_COLUMN,
    MONTHS,
    RAW_CATEGORICAL_INPUTS,
    SPLITS,
    TARGET,
    TEST_END,
    TIME_COLUMN,
    TRAIN_END,
)
from hotel_booking.data import read_split

HOTELS = ["City Hotel", "Resort Hotel"]
MAX_ADR = 1000
MAX_CHILDREN = 5
REPORT_FAILURE_CASES = 20


def _numeric(*checks: pa.Check, nullable: bool = False) -> pa.Column:
    """A numeric column, coerced to float so integer and float CSV columns are checked alike.

    Args:
        *checks: Value checks for the column.
        nullable: Whether missing values are allowed.

    Returns:
        The pandera column.
    """
    return pa.Column(float, list(checks), nullable=nullable, coerce=True)


def _time_window(split: str) -> pa.Check:
    """Check that every arrival date falls inside the window of `split`.

    Args:
        split: Split name, one of `SPLITS`.

    Returns:
        A check on the arrival date column.
    """
    train_end, test_end = pd.Timestamp(TRAIN_END), pd.Timestamp(TEST_END)
    windows = {
        "train": lambda dates: dates <= train_end,
        "test": lambda dates: (dates > train_end) & (dates <= test_end),
        "production_holdout": lambda dates: dates > test_end,
    }
    return pa.Check(windows[split], name=f"inside_{split}_window")


def split_schema(split: str | None = None) -> pa.DataFrameSchema:
    """Schema of one processed split; with `split`, arrival dates must also fall inside that split's window.

    The schema is strict: any extra column, including the leakage columns, fails validation.

    Args:
        split: Split name, one of `SPLITS`, or `None` to skip the time window check.

    Returns:
        The pandera schema.
    """
    dates = [_time_window(split)] if split else []
    categorical = {column: pa.Column(str) for column in RAW_CATEGORICAL_INPUTS}
    categorical["hotel"] = pa.Column(str, pa.Check.isin(HOTELS))
    categorical["arrival_date_month"] = pa.Column(str, pa.Check.isin(MONTHS))
    columns = {
        ID_COLUMN: pa.Column(int, pa.Check.ge(0), unique=True),
        TIME_COLUMN: pa.Column("datetime64[ns]", dates, coerce=True),
        **categorical,
        "lead_time": _numeric(pa.Check.ge(0)),
        "arrival_date_year": _numeric(pa.Check.in_range(2015, 2017)),
        "arrival_date_week_number": _numeric(pa.Check.in_range(1, 53)),
        "arrival_date_day_of_month": _numeric(pa.Check.in_range(1, 31)),
        "stays_in_weekend_nights": _numeric(pa.Check.ge(0)),
        "stays_in_week_nights": _numeric(pa.Check.ge(0)),
        "adults": _numeric(pa.Check.ge(0)),
        "children": _numeric(pa.Check.in_range(0, MAX_CHILDREN), nullable=True),
        "babies": _numeric(pa.Check.in_range(0, MAX_CHILDREN)),
        "is_repeated_guest": _numeric(pa.Check.isin([0, 1])),
        "previous_cancellations": _numeric(pa.Check.ge(0)),
        "previous_bookings_not_canceled": _numeric(pa.Check.ge(0)),
        "agent": _numeric(pa.Check.ge(0), nullable=True),
        "company": _numeric(pa.Check.ge(0), nullable=True),
        "days_in_waiting_list": _numeric(pa.Check.ge(0)),
        "adr": _numeric(pa.Check.in_range(0, MAX_ADR)),
        "total_of_special_requests": _numeric(pa.Check.ge(0)),
        GROUP_SIZE_COLUMN: _numeric(pa.Check.ge(1)),
        "assigned_room_type": pa.Column(str),
        "booking_changes": _numeric(pa.Check.ge(0)),
        "required_car_parking_spaces": _numeric(pa.Check.ge(0)),
        "country": pa.Column(str, nullable=True),
        TARGET: _numeric(pa.Check.isin([0, 1])),
    }
    has_guests = pa.Check(
        lambda frame: frame["adults"] + frame["children"].fillna(0) + frame["babies"] > 0, name="has_guests"
    )
    return pa.DataFrameSchema(columns, checks=[has_guests], strict=True, name=split or "processed")


def validate_split(frame: pd.DataFrame, split: str | None = None) -> pd.DataFrame:
    """Return the failures of `frame` against the split schema, one row per failing value.

    Args:
        frame: A processed split.
        split: Split name, one of `SPLITS`, or `None` to skip the time window check.

    Returns:
        Failure cases with `column`, `check` and `failure_case`; an empty frame means the split is valid.
    """
    try:
        split_schema(split).validate(frame, lazy=True)
    except pa.errors.SchemaErrors as error:
        cases = error.failure_cases[["column", "check", "failure_case"]]
        return cases.astype({"column": "str", "check": "str", "failure_case": "str"}).reset_index(drop=True)
    return pd.DataFrame(columns=["column", "check", "failure_case"])


def summarize(splits: dict[str, pd.DataFrame]) -> dict[str, dict[str, object]]:
    """Validate every split and summarize the result.

    Args:
        splits: Processed splits keyed by name.

    Returns:
        Per split: row count, number of failures, whether it passed and the first failure cases.
    """
    summary: dict[str, dict[str, object]] = {}
    for name, frame in splits.items():
        failures = validate_split(frame, name)
        summary[name] = {
            "rows": len(frame),
            "failures": len(failures),
            "passed": failures.empty,
            "failure_cases": failures.head(REPORT_FAILURE_CASES).to_dict(orient="records"),
        }
    return summary


def main(processed_dir: Path, report_path: Path) -> None:
    """Validate every split in `processed_dir`, write a JSON report and exit with code 1 if any split fails.

    Args:
        processed_dir: Folder written by `data.preprocess`.
        report_path: Where to write the JSON report.
    """
    summary = summarize({name: read_split(processed_dir, name) for name in SPLITS})
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(f"{json.dumps(summary, indent=2, ensure_ascii=False)}\n", encoding="utf-8")
    for name, result in summary.items():
        print(f"{name}: {result['rows']} rows, {result['failures']} failures")
    if not all(result["passed"] for result in summary.values()):
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
