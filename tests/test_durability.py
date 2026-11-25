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
