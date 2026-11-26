import json
from pathlib import Path

import pandas as pd
import pytest
import typer

from hotel_booking import config, data, validation


@pytest.fixture(scope="module")
def processed(raw_bookings: pd.DataFrame, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Processed splits of the synthetic bookings, written and read back like the real pipeline does."""
    root = tmp_path_factory.mktemp("validation")
    (root / "raw").mkdir()
    raw_bookings.to_csv(root / "raw" / config.RAW_FILE, index=False)
    data.preprocess(root / "raw", root / "processed")
    return root / "processed"


def test_processed_splits_pass(processed: Path) -> None:
    """Every split written by preprocess passes its own schema, time window included."""
    for name in config.SPLITS:
        assert validation.validate_split(data.read_split(processed, name), name).empty


def test_leakage_column_and_bad_values_fail(processed: Path) -> None:
    """A leakage column, an out-of-range adr and an unknown hotel are all reported in one pass."""
    train = data.read_split(processed, "train").head(20).copy()
    train["reservation_status"] = "Canceled"
    train.loc[0, "adr"] = 5000.0
    train.loc[1, "hotel"] = "Unknown Hotel"
    failures = validation.validate_split(train, "train")
    assert {"adr", "hotel"} <= set(failures["column"])
    extra = failures[failures["check"] == "column_in_schema"]
    assert extra["failure_case"].tolist() == ["reservation_status"]


def test_rows_outside_split_window_fail(processed: Path) -> None:
    """A 2016 booking in the test split breaks the temporal split."""
    test = data.read_split(processed, "test").head(5).copy()
    test.loc[0, config.TIME_COLUMN] = pd.Timestamp("2016-06-01")
    failures = validation.validate_split(test, "test")
    assert failures["check"].str.contains("inside_test_window").any()


def test_duplicate_ids_and_rows_without_guests_fail(processed: Path) -> None:
    """Booking ids must be unique and every booking must have at least one guest."""
    train = data.read_split(processed, "train").head(10).copy()
    ids = train[config.ID_COLUMN].to_numpy().copy()
    ids[1] = ids[0]
    train[config.ID_COLUMN] = ids
    train.loc[2, ["adults", "children", "babies"]] = 0
    checks = set(validation.validate_split(train, "train")["check"])
    assert {"field_uniqueness", "has_guests"} <= checks


def test_main_writes_report_and_fails_on_bad_split(processed: Path, tmp_path: Path) -> None:
    """The CLI writes a JSON report and exits with code 1 when one split is invalid."""
    report = tmp_path / "validation.json"
    validation.main(processed, report)
    assert all(result["passed"] for result in json.loads(report.read_text(encoding="utf-8")).values())

    broken = tmp_path / "broken"
    broken.mkdir()
    for name in config.SPLITS:
        frame = pd.read_csv(processed / f"{name}.csv")
        if name == "train":
            frame.loc[0, "adr"] = -10
        frame.to_csv(broken / f"{name}.csv", index=False)
    with pytest.raises(typer.Exit) as exit_info:
        validation.main(broken, report)
    assert exit_info.value.exit_code == 1
    summary = json.loads(report.read_text(encoding="utf-8"))
    assert not summary["train"]["passed"] and summary["test"]["passed"]


def test_real_processed_splits_pass(real_raw: pd.DataFrame) -> None:
    """On the real dataset every split passes validation."""
    clean, _ = data.clean_bookings(real_raw)
    for name, frame in data.time_split(clean).items():
        assert validation.validate_split(frame, name).empty, name
