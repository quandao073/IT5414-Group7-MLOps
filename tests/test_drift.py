import json
from pathlib import Path

import pandas as pd
import pytest

from hotel_booking import config, data, drift


@pytest.fixture(scope="module")
def processed(raw_bookings: pd.DataFrame, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Processed splits of the synthetic bookings."""
    root = tmp_path_factory.mktemp("drift")
    (root / "raw").mkdir()
    raw_bookings.to_csv(root / "raw" / config.RAW_FILE, index=False)
    data.preprocess(root / "raw", root / "processed")
    return root / "processed"


def test_drift_columns_skip_calendar_and_excluded() -> None:
    """Calendar columns and excluded columns are never compared; agent and company are categories."""
    numerical, categorical = drift.drift_columns(exclude=("hotel",))
    compared = set(numerical) | set(categorical)
    assert not compared & set(drift.CALENDAR_COLUMNS) and "hotel" not in compared
    assert {"agent", "company"} <= set(categorical) and "lead_time" in numerical


def test_shifted_column_drifts_and_same_data_does_not(sample: pd.DataFrame) -> None:
    """Shifting lead_time makes it drift; comparing the data with itself flags nothing."""
    _, same = drift.column_drift(sample, sample)
    assert not same["drifted"].any()
    _, shifted = drift.column_drift(sample, sample.assign(lead_time=sample["lead_time"] + 500))
    assert shifted.set_index("column").loc["lead_time", "drifted"]


def test_scenarios_cover_2017_months_and_hotels(processed: Path) -> None:
    """Every 2017 month in the stream gets a scenario, plus City Hotel against Resort Hotel."""
    result = drift.scenarios(processed)
    assert drift.HOTEL_SCENARIO in result
    months = [name for name in result if name != drift.HOTEL_SCENARIO]
    assert months and all(name.startswith("2017-") for name in months)
    reference, current, exclude = result[drift.HOTEL_SCENARIO]
    assert set(reference["hotel"]) == {"City Hotel"} and set(current["hotel"]) == {"Resort Hotel"}
    assert exclude == ("hotel",)


def test_main_writes_reports_and_summary(processed: Path, tmp_path: Path) -> None:
    """The CLI writes one HTML report per scenario and consistent summary files."""
    drift.main(processed, tmp_path)
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert set(summary) == set(drift.scenarios(processed))
    for name, row in summary.items():
        assert (tmp_path / "html" / f"{name}.html").stat().st_size > 0
        assert 0 <= row["drift_share"] <= 1 and row["drifted_columns"] <= row["columns"]
    columns = pd.read_csv(tmp_path / "columns.csv")
    assert set(columns["scenario"]) == set(summary)
