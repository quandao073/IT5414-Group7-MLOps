"""Simulate production: replay the 2017 bookings month by month into the API (contract C4 in `docs/contracts.md`).

The labels of a month are released only after every booking of that month has been sent, to mimic the delay
between a prediction and knowing whether the booking was canceled.
"""

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import typer

from hotel_booking.config import ID_COLUMN, MODEL_INPUT_COLUMNS, TARGET, TIME_COLUMN
from hotel_booking.data import read_split

STREAM_SPLITS = ("test", "production_holdout")
LABEL_COLUMNS = [ID_COLUMN, TARGET]
LABELS_DIR = "labels"


def month_key(dates: pd.Series) -> pd.Series:
    """Arrival month as `YYYY-MM`.

    Args:
        dates: Arrival dates.

    Returns:
        The month of each date.
    """
    return dates.dt.strftime("%Y-%m")


def build_stream(processed_dir: Path, stream_dir: Path) -> list[Path]:
    """Write one CSV per arrival month from the test and production holdout splits, keeping existing files.

    Args:
        processed_dir: Folder with the splits from `invoke preprocess-data`.
        stream_dir: Folder for the monthly batches `YYYY-MM.csv`.

    Returns:
        The monthly batch files, sorted by month.
    """
    stream_dir.mkdir(parents=True, exist_ok=True)
    stream = pd.concat([read_split(processed_dir, name) for name in STREAM_SPLITS], ignore_index=True)
    for month, batch in stream.groupby(month_key(stream[TIME_COLUMN]), sort=True):
        path = stream_dir / f"{month}.csv"
        if not path.exists():
            batch.to_csv(path, index=False)
    return list_batches(stream_dir)


def list_batches(stream_dir: Path, months: list[str] | None = None) -> list[Path]:
    """Monthly batch files in the stream folder.

    Args:
        stream_dir: Folder with `YYYY-MM.csv` files.
        months: Only these months, e.g. `["2017-01"]`; all when `None`.

    Returns:
        The batch files, sorted by month.
    """
    paths = sorted(stream_dir.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9].csv"))
    return [path for path in paths if months is None or path.stem in months]


def to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Model input columns as JSON-ready records, with missing values as `None`.

    Args:
        frame: Bookings.

    Returns:
        One dict per booking.
    """
    records: list[dict[str, Any]] = json.loads(frame[MODEL_INPUT_COLUMNS].to_json(orient="records"))
    return records


def release_labels(batch: pd.DataFrame, month: str, labels_dir: Path) -> Path:
    """Write the labels of a replayed month.

    Args:
        batch: The bookings of the month, with the target.
        month: The month, `YYYY-MM`.
        labels_dir: Folder for the label files.

    Returns:
        The label file.
    """
    labels_dir.mkdir(parents=True, exist_ok=True)
    labels = batch[LABEL_COLUMNS].assign(month=month, released_at=datetime.now(UTC).isoformat(timespec="seconds"))
    path = labels_dir / f"{month}.csv"
    labels.to_csv(path, index=False)
    return path


def replay_month(
    client: httpx.Client, batch_file: Path, labels_dir: Path, batch_size: int, delay: float
) -> dict[str, Any]:
    """Send one month to `POST /predict/batch`, then release its labels.

    Args:
        client: HTTP client pointing at the API.
        batch_file: The month's `YYYY-MM.csv`.
        labels_dir: Folder for the label files.
        batch_size: Bookings per request.
        delay: Seconds to wait between requests.

    Returns:
        Summary of the month: rows, requests, mean probability, model versions seen and label file.
    """
    batch = read_split(batch_file.parent, batch_file.stem)
    probabilities: list[float] = []
    versions: set[str] = set()
    requests = 0
    for start in range(0, len(batch), batch_size):
        if requests and delay:
            time.sleep(delay)
        chunk = batch.iloc[start : start + batch_size]
        response = client.post("/predict/batch", json={"bookings": to_records(chunk)})
        response.raise_for_status()
        body = response.json()
        probabilities += [item["probability"] for item in body["predictions"]]
        versions.add(body["model_version"])
        requests += 1
    labels = release_labels(batch, batch_file.stem, labels_dir)
    return {
        "month": batch_file.stem,
        "rows": len(batch),
        "requests": requests,
        "mean_probability": round(sum(probabilities) / max(len(probabilities), 1), 4),
        "model_versions": sorted(versions),
        "labels": str(labels),
    }


def main(
    api_url: str = "http://localhost:8000",
    processed_dir: Path = Path("data/processed"),
    stream_dir: Path = Path("data/stream"),
    months: str = "",
    batch_size: int = 200,
    delay: float = 0.0,
) -> None:
    """Replay the 2017 stream into the API month by month.

    Args:
        api_url: Base URL of the API.
        processed_dir: Folder with the splits, used to build the stream when it does not exist yet.
        stream_dir: Folder with the monthly batches; labels go to its `labels/` subfolder.
        months: Comma-separated months to replay, e.g. `2017-01,2017-02`; all months when empty.
        batch_size: Bookings per request.
        delay: Seconds to wait between requests, to control the replay speed.
    """
    selected = [month.strip() for month in months.split(",") if month.strip()] or None
    batches = list_batches(stream_dir, selected)
    if not batches:
        build_stream(processed_dir, stream_dir)
        batches = list_batches(stream_dir, selected)
    with httpx.Client(base_url=api_url, timeout=60) as client:
        for batch_file in batches:
            print(json.dumps(replay_month(client, batch_file, stream_dir / LABELS_DIR, batch_size, delay)))


if __name__ == "__main__":
    typer.run(main)
