"""Load, clean and split the bookings, and validate new inputs against a schema learned from train."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer

from hotel_booking.config import (
    GROUP_SIZE_COLUMN,
    ID_COLUMN,
    INPUT_COLUMNS,
    LEAKAGE_COLUMNS,
    MONTHS,
    POST_BOOKING_COLUMNS,
    RAW_CATEGORICAL_INPUTS,
    RAW_FILE,
    RAW_NUMERIC_INPUTS,
    SPLITS,
    TARGET,
    TEST_END,
    TIME_COLUMN,
    TRAIN_END,
)

MONTH_TO_NUMBER = {month: number for number, month in enumerate(MONTHS, start=1)}
KAGGLE_URL = "https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand"
SCHEMA_FILE = "input_schema.json"

# Numeric columns without a range check: the arrival year grows over time, agent/company are identifiers.
RANGE_EXEMPT = {"arrival_date_year", "agent", "company"}

Schema = dict[str, dict[str, Any]]
Fold = tuple[np.ndarray, np.ndarray]


def load_raw(path: Path) -> pd.DataFrame:
    """Read the raw Kaggle CSV.

    Args:
        path: Path to `hotel_bookings.csv`.

    Returns:
        The raw bookings.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `dvc pull` or download {RAW_FILE} from {KAGGLE_URL}.")
    return pd.read_csv(path)


def arrival_dates(frame: pd.DataFrame) -> pd.Series:
    """Build the arrival date from the year, month name and day-of-month columns.

    Args:
        frame: Bookings with the raw arrival date columns.

    Returns:
        Arrival dates as a datetime series.
    """
    parts = pd.DataFrame(
        {
            "year": frame["arrival_date_year"],
            "month": frame["arrival_date_month"].map(MONTH_TO_NUMBER),
            "day": frame["arrival_date_day_of_month"],
        }
    )
    return pd.to_datetime(parts)


def add_group_size(frame: pd.DataFrame) -> pd.Series:
    """Count the rows that share every booking-time detail, i.e. the rooms of one group booking.

    Only booking-time inputs are used, never the target, so the count does not leak the outcome.

    Args:
        frame: Bookings with all columns in `INPUT_COLUMNS`.

    Returns:
        The group size of each row.
    """
    keys = frame[INPUT_COLUMNS].assign(_row=1)
    return keys.groupby(INPUT_COLUMNS, sort=False, dropna=False)["_row"].transform("size").astype(int)


def clean_bookings(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop invalid rows and leakage columns, keeping raw values so the pipeline builds the features.

    Args:
        frame: The raw bookings.

    Returns:
        The cleaned bookings and a log of how many rows each rule removed.
    """
    log = []
    out = frame.copy()
    out.insert(0, ID_COLUMN, np.arange(len(out)))

    def drop(mask: pd.Series, reason: str) -> None:
        nonlocal out
        log.append({"step": reason, "rows_removed": int(mask.sum())})
        out = out.loc[~mask]

    guests = out["adults"] + out["children"].fillna(0) + out["babies"]
    drop(guests == 0, "no guests")
    drop((out["adr"] < 0) | (out["adr"] > 1000), "adr negative or above 1000")
    drop((out["children"].fillna(0) > 5) | (out["babies"] > 5), "more than 5 children or babies")

    out = out.drop(columns=LEAKAGE_COLUMNS)
    out.insert(1, TIME_COLUMN, arrival_dates(out))
    out[GROUP_SIZE_COLUMN] = add_group_size(out)
    columns = [ID_COLUMN, TIME_COLUMN] + INPUT_COLUMNS + [GROUP_SIZE_COLUMN] + POST_BOOKING_COLUMNS + [TARGET]
    return out[columns].reset_index(drop=True), pd.DataFrame(log)


def time_split(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split the cleaned bookings by arrival date into train, test and production holdout.

    Args:
        frame: The cleaned bookings.

    Returns:
        One frame per split name in `SPLITS`, sorted by arrival date.
    """
    dates = frame[TIME_COLUMN]
    period = np.select([dates <= TRAIN_END, dates <= TEST_END], ["train", "test"], default="production_holdout")
    return {name: frame[period == name].sort_values([TIME_COLUMN, ID_COLUMN]).reset_index(drop=True) for name in SPLITS}


def read_split(processed_dir: Path, name: str) -> pd.DataFrame:
    """Read one split written by `preprocess`, keeping categorical codes as strings.

    Args:
        processed_dir: Folder that holds the split CSV files.
        name: Split name, one of `SPLITS`.

    Returns:
        The split, sorted by arrival date.
    """
    path = processed_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `uv run invoke preprocess-data` first.")
    dtypes = {column: "str" for column in RAW_CATEGORICAL_INPUTS + ["assigned_room_type", "country"]}
    frame = pd.read_csv(path, dtype=dtypes, parse_dates=[TIME_COLUMN])
    return frame.sort_values([TIME_COLUMN, ID_COLUMN]).reset_index(drop=True)


def quarterly_folds(dates: pd.Series, year: int = 2016) -> tuple[list[Fold], pd.DataFrame]:
    """Expanding-window folds: each quarter of `year` is validated on a model trained on everything before it.

    Args:
        dates: Arrival dates of the training rows, in row order.
        year: The year whose quarters become the validation folds.

    Returns:
        The (train indices, validation indices) of each fold and a table describing the folds.
    """
    folds, rows = [], []
    for quarter in range(1, 5):
        start = pd.Timestamp(year=year, month=3 * quarter - 2, day=1)
        end = start + pd.offsets.QuarterEnd(0)
        train_idx = np.flatnonzero(dates < start)
        valid_idx = np.flatnonzero((dates >= start) & (dates <= end))
        folds.append((train_idx, valid_idx))
        rows.append(
            {
                "fold": f"{year}Q{quarter}",
                "train_rows": len(train_idx),
                "valid_from": f"{start:%Y-%m-%d}",
                "valid_to": f"{end:%Y-%m-%d}",
                "valid_rows": len(valid_idx),
            }
        )
    return folds, pd.DataFrame(rows)


def recency_weights(dates: pd.Series, half_life_days: float | None) -> np.ndarray:
    """Sample weights that halve every `half_life_days` days before the latest date.

    Args:
        dates: Arrival dates.
        half_life_days: Half-life in days; `None` or 0 gives equal weights.

    Returns:
        One weight per date, 1.0 for the latest date.
    """
    if not half_life_days:
        return np.ones(len(dates))
    age = (dates.max() - dates).dt.days.to_numpy()
    return np.power(0.5, age / half_life_days)


def build_input_schema(train: pd.DataFrame) -> Schema:
    """Learn the input schema from train: allowed categories and numeric ranges.

    Args:
        train: The training split.

    Returns:
        A schema with a `categorical` and a `numeric` section.
    """
    schema: Schema = {"categorical": {}, "numeric": {}}
    for column in RAW_CATEGORICAL_INPUTS:
        schema["categorical"][column] = sorted(train[column].dropna().unique().tolist())
    for column in RAW_NUMERIC_INPUTS + [GROUP_SIZE_COLUMN]:
        values = train[column]
        exempt = column in RANGE_EXEMPT
        schema["numeric"][column] = {
            "min": None if exempt else float(values.min()),
            "max": None if exempt else float(values.max()),
            "nullable": bool(values.isna().any()),
        }
    return schema


def validate_inputs(frame: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    """Check new bookings against the schema.

    Args:
        frame: Bookings to check.
        schema: Schema from `build_input_schema`.

    Returns:
        One row per problem (column, issue, affected rows); an empty frame means the input is valid.
    """
    issues = []
    for column in list(schema["categorical"]) + list(schema["numeric"]):
        if column not in frame.columns:
            issues.append({"column": column, "issue": "missing column", "rows": len(frame)})
    for column, allowed in schema["categorical"].items():
        if column in frame.columns:
            unknown = ~frame[column].isin(allowed) & frame[column].notna()
            if unknown.any():
                issues.append({"column": column, "issue": "unseen category", "rows": int(unknown.sum())})
    for column, rule in schema["numeric"].items():
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        not_numeric = values.isna() & frame[column].notna()
        if not_numeric.any():
            issues.append({"column": column, "issue": "not numeric", "rows": int(not_numeric.sum())})
        if not rule["nullable"] and values.isna().any():
            issues.append({"column": column, "issue": "missing value", "rows": int(values.isna().sum())})
        if rule["min"] is None:
            continue
        outside = (values < rule["min"]) | (values > rule["max"])
        if outside.any():
            issues.append({"column": column, "issue": "outside train range", "rows": int(outside.sum())})
    return pd.DataFrame(issues, columns=["column", "issue", "rows"])


def preprocess(data_path: Path, output_folder: Path) -> None:
    """Clean the raw CSV, split it by arrival date and write the splits plus the input schema.

    Args:
        data_path: Folder that holds `hotel_bookings.csv`.
        output_folder: Folder for `train.csv`, `test.csv`, `production_holdout.csv` and `input_schema.json`.
    """
    clean, log = clean_bookings(load_raw(data_path / RAW_FILE))
    splits = time_split(clean)
    output_folder.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_csv(output_folder / f"{name}.csv", index=False)
    schema = build_input_schema(splits["train"])
    (output_folder / SCHEMA_FILE).write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(log.to_string(index=False))
    print({name: len(frame) for name, frame in splits.items()})


if __name__ == "__main__":
    typer.run(preprocess)
