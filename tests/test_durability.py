import numpy as np
import pandas as pd
import pytest

from hotel_booking import config, durability


class _LeadTimeModel:
    """Rank by lead_time; only the column the tests set is read."""

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        score = np.clip(frame["lead_time"].to_numpy() / 400, 0, 1)
        return np.column_stack([1 - score, score])


@pytest.fixture
def two_months(sample: pd.DataFrame) -> pd.DataFrame:
    """Two labelled months built from the cleaned synthetic bookings."""
    frame = sample.iloc[:160].copy()
    frame[config.TIME_COLUMN] = pd.to_datetime(np.where(np.arange(len(frame)) < 80, "2017-01-15", "2017-02-15"))
    frame[config.TARGET] = (frame["lead_time"] > frame["lead_time"].median()).astype(int)
    return frame


def test_bootstrap_roc_auc_is_deterministic_and_bounded() -> None:
    """The interval is reproducible, contains the point estimate and stays within [0, 1]."""
    y = [0, 0, 1, 1, 0, 1, 1, 0]
    proba = [0.1, 0.4, 0.6, 0.9, 0.2, 0.7, 0.8, 0.3]
    score, low, high = durability.bootstrap_roc_auc(y, proba, rounds=200)
    assert (score, low, high) == durability.bootstrap_roc_auc(y, proba, rounds=200)
    assert 0 <= low <= score <= high <= 1


def test_bootstrap_without_rounds_returns_the_point_estimate() -> None:
    """Zero rounds means no resampling, so the interval collapses to the point estimate."""
    score, low, high = durability.bootstrap_roc_auc([0, 1], [0.2, 0.8], rounds=0)
    assert score == low == high == 1.0


def test_monthly_metrics_one_row_per_month_sorted(two_months: pd.DataFrame) -> None:
    """Each month gets one row with its size and cancel rate, in calendar order."""
    table = durability.monthly_metrics(_LeadTimeModel(), two_months)
    assert table["month"].tolist() == ["2017-01", "2017-02"]
    assert table["rows"].tolist() == [80, 80]
    assert table["cancel_rate"].between(0, 1).all()
    assert table["rows"].sum() == len(two_months)


def test_monthly_metrics_skips_single_class_month(two_months: pd.DataFrame) -> None:
    """A month with one label has no ROC-AUC and is left out instead of crashing."""
    frame = two_months.copy()
    frame.loc[frame[config.TIME_COLUMN].dt.month == 2, config.TARGET] = 0
    table = durability.monthly_metrics(_LeadTimeModel(), frame)
    assert table["month"].tolist() == ["2017-01"]


def test_monthly_metrics_combines_rows_of_the_same_month(two_months: pd.DataFrame) -> None:
    """Two chunks with the same month are one row whose size is their sum."""
    table = durability.monthly_metrics(_LeadTimeModel(), pd.concat([two_months, two_months]))
    assert table["rows"].tolist() == [160, 160]


FAST_SPEC = {
    "name": "Logistic Regression",
    "encoding": "onehot",
    "half_life_days": None,
    "booking_weights": False,
    "pipeline": {},
    "params": {"max_iter": 200},
}


@pytest.fixture
def hotel_frame(sample: pd.DataFrame) -> pd.DataFrame:
    """Both hotels and both labels in every subset, so ROC-AUC is always defined."""
    frame = sample.iloc[:400].copy()
    frame["hotel"] = np.where(np.arange(len(frame)) % 2 == 0, "City Hotel", "Resort Hotel")
    frame[config.TARGET] = (np.arange(len(frame)) // 2) % 2
    return frame


def test_hotel_transfer_has_the_four_combinations(hotel_frame: pd.DataFrame) -> None:
    """Every source/target pair is scored, the diagonal is in-domain and has no gap."""
    table = durability.hotel_transfer_metrics(hotel_frame, hotel_frame, FAST_SPEC)
    pairs = set(zip(table["source_hotel"], table["target_hotel"], strict=True))
    assert pairs == {(source, target) for source in durability.HOTELS for target in durability.HOTELS}
    diagonal = table[table["in_domain"]]
    assert len(diagonal) == 2 and (diagonal["transfer_gap"] == 0).all()
    assert table["roc_auc"].between(0, 1).all()


def test_hotel_transfer_skips_a_hotel_with_one_class(hotel_frame: pd.DataFrame) -> None:
    """A degenerate hotel is left out; the remaining pairs still produce a table."""
    frame = hotel_frame.copy()
    frame.loc[frame["hotel"] == "Resort Hotel", config.TARGET] = 0
    table = durability.hotel_transfer_metrics(frame, frame, FAST_SPEC)
    assert set(table["source_hotel"]) == {"City Hotel"}


def test_hotel_transfer_handles_a_category_seen_in_one_hotel_only(hotel_frame: pd.DataFrame) -> None:
    """A category absent from the source hotel must not break prediction on the target hotel."""
    frame = hotel_frame.copy()
    city = frame["hotel"] == "City Hotel"
    frame.loc[city & (frame["meal"] == "HB"), "meal"] = "SC"
    table = durability.hotel_transfer_metrics(frame, frame, FAST_SPEC)
    assert len(table) == 4
