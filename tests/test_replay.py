from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from hotel_booking import config, data, replay, sample_model
from hotel_booking.api import create_app

SPEC = {"name": "Logistic Regression", "params": {"C": 0.1}}


@pytest.fixture(scope="module")
def processed_dir(raw_bookings: pd.DataFrame, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Splits written by `preprocess` from synthetic raw bookings."""
    root = tmp_path_factory.mktemp("replay")
    (root / "raw").mkdir()
    raw_bookings.to_csv(root / "raw" / config.RAW_FILE, index=False)
    data.preprocess(root / "raw", root / "processed")
    return root / "processed"


def test_build_stream_writes_one_file_per_2017_month(processed_dir: Path, tmp_path: Path) -> None:
    """The stream has one batch per month from 2017-01 to 2017-08, covering test and production holdout."""
    batches = replay.build_stream(processed_dir, tmp_path / "stream")
    assert [path.stem for path in batches] == [f"2017-{month:02d}" for month in range(1, 9)]
    expected = sum(len(data.read_split(processed_dir, name)) for name in replay.STREAM_SPLITS)
    assert sum(len(data.read_split(path.parent, path.stem)) for path in batches) == expected


def test_list_batches_filters_months(processed_dir: Path, tmp_path: Path) -> None:
    """Only the requested months are replayed."""
    replay.build_stream(processed_dir, tmp_path)
    assert [path.stem for path in replay.list_batches(tmp_path, ["2017-03", "2017-01"])] == ["2017-01", "2017-03"]


def test_replay_month_predicts_every_row_then_releases_labels(processed_dir: Path, tmp_path: Path) -> None:
    """A replayed month sends every booking to the API and writes its labels afterwards."""
    splits = {name: data.read_split(processed_dir, name) for name in ("train", "test")}
    model, card = sample_model.train_with_card(SPEC, splits["train"], splits["test"], "LogReg", "replay-test")
    sample_model.save_artifacts(model, card, tmp_path / "models")
    batch_file = replay.build_stream(processed_dir, tmp_path / "stream")[0]

    app = create_app(tmp_path / "models" / sample_model.MODEL_FILE, tmp_path / "models" / sample_model.CARD_FILE)
    with TestClient(app) as client:
        summary = replay.replay_month(client, batch_file, tmp_path / "labels", batch_size=7, delay=0)

    rows = len(data.read_split(batch_file.parent, batch_file.stem))
    assert summary["rows"] == rows and summary["requests"] == -(-rows // 7)
    assert summary["model_versions"] == ["replay-test"]
    labels = pd.read_csv(summary["labels"])
    assert len(labels) == rows and set(labels["month"]) == {batch_file.stem}
    assert set(labels.columns) == {config.ID_COLUMN, config.TARGET, "month", "released_at"}
