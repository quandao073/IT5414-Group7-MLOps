import json
from pathlib import Path

import numpy as np
import pandas as pd

from hotel_booking import config, data


def test_clean_removes_leakage_and_invalid_rows(raw_bookings: pd.DataFrame, sample: pd.DataFrame) -> None:
    """Cleaning drops the leakage columns and rows without guests, and keeps unique booking ids."""
    assert not set(config.LEAKAGE_COLUMNS) & set(sample.columns)
    assert sample[config.ID_COLUMN].is_unique
    guests = sample["adults"] + sample["children"].fillna(0) + sample["babies"]
    assert (guests > 0).all()
    assert len(sample) == len(raw_bookings) - 1


def test_group_size_ignores_target() -> None:
    """Rows that differ only in the target belong to the same group booking."""
    row: dict[str, object] = {column: 1 for column in config.RAW_NUMERIC_INPUTS}
    row.update({column: "x" for column in config.RAW_CATEGORICAL_INPUTS})
    row.update({"agent": np.nan, "company": np.nan})
    frame = pd.DataFrame([row, row, {**row, "lead_time": 2}])
    frame[config.TARGET] = [0, 1, 1]
    assert data.add_group_size(frame).tolist() == [2, 2, 1]


def test_time_split_is_ordered_and_complete(sample: pd.DataFrame) -> None:
    """Every row lands in exactly one split and the splits do not overlap in time."""
    splits = data.time_split(sample)
    assert sum(len(frame) for frame in splits.values()) == len(sample)
    assert splits["train"][config.TIME_COLUMN].max() < splits["test"][config.TIME_COLUMN].min()
    assert splits["test"][config.TIME_COLUMN].max() < splits["production_holdout"][config.TIME_COLUMN].min()


def test_quarterly_folds_never_train_on_future(sample: pd.DataFrame) -> None:
    """Each fold trains only on bookings that arrive before its validation quarter."""
    train = data.time_split(sample)["train"]
    folds, table = data.quarterly_folds(train[config.TIME_COLUMN])
    assert len(folds) == 4 and (table["valid_rows"] > 0).all()
    for train_idx, valid_idx in folds:
        assert train[config.TIME_COLUMN].iloc[train_idx].max() < train[config.TIME_COLUMN].iloc[valid_idx].min()


def test_recency_weights_halve_every_half_life() -> None:
    """The latest date weighs 1 and a date one half-life earlier weighs about 0.5."""
    dates = pd.Series(pd.to_datetime(["2016-01-01", "2016-07-01", "2016-12-31"]))
    weights = data.recency_weights(dates, half_life_days=365)
    assert weights[-1] == 1.0
    assert np.isclose(weights[0], 0.5, atol=0.01)
    assert (data.recency_weights(dates, None) == 1).all()


def test_schema_accepts_train_and_flags_bad_rows(sample: pd.DataFrame) -> None:
    """Train passes its own schema; unseen categories, missing columns and text in numbers are flagged."""
    train = data.time_split(sample)["train"]
    schema = data.build_input_schema(train)
    assert data.validate_inputs(train, schema).empty
    bad = train.head(3).copy()
    bad.loc[bad.index[0], "hotel"] = "Unknown Hotel"
    bad["lead_time"] = bad["lead_time"].astype(object)
    bad.loc[bad.index[1], "lead_time"] = "abc"
    bad = bad.drop(columns="adr")
    issues = set(data.validate_inputs(bad, schema)["issue"])
    assert {"unseen category", "missing column", "not numeric"} <= issues


def test_schema_does_not_flag_future_years_or_new_ids(sample: pd.DataFrame) -> None:
    """Arrival year and agent/company ids are exempt from the train range check."""
    splits = data.time_split(sample)
    schema = data.build_input_schema(splits["train"])
    later = splits["production_holdout"].assign(agent=99999.0)
    assert not set(data.validate_inputs(later, schema)["column"]) & data.RANGE_EXEMPT


def test_preprocess_writes_splits_and_schema(raw_bookings: pd.DataFrame, tmp_path: Path) -> None:
    """Preprocess writes one readable CSV per split and the input schema."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    raw_bookings.to_csv(raw_dir / config.RAW_FILE, index=False)
    data.preprocess(raw_dir, tmp_path / "processed")
    for name in config.SPLITS:
        split = data.read_split(tmp_path / "processed", name)
        assert len(split) > 0 and split[config.TIME_COLUMN].is_monotonic_increasing
    schema = json.loads((tmp_path / "processed" / data.SCHEMA_FILE).read_text(encoding="utf-8"))
    assert set(schema["numeric"]) == set(config.RAW_NUMERIC_INPUTS + [config.GROUP_SIZE_COLUMN])


def test_real_dataset_cleaning_and_split(real_raw: pd.DataFrame) -> None:
    """On the real dataset, cleaning removes 185 rows and the split sizes match the IT3190 model card."""
    clean, log = data.clean_bookings(real_raw)
    assert log["rows_removed"].sum() == 185
    sizes = {name: len(frame) for name, frame in data.time_split(clean).items()}
    assert sizes == {"train": 78586, "test": 18442, "production_holdout": 22177}
