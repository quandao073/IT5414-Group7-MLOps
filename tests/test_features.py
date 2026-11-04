from typing import Any

import numpy as np
import pandas as pd

from hotel_booking import config
from hotel_booking.features import BookingFeatureBuilder


def test_output_columns_are_fixed(sample: pd.DataFrame) -> None:
    """Output columns follow the declared feature order and numeric features have no missing values."""
    builder = BookingFeatureBuilder().fit(sample)
    features = builder.transform(sample[config.MODEL_INPUT_COLUMNS])
    assert list(features.columns) == builder.numeric_features() + builder.categorical_features()
    assert list(builder.get_feature_names_out()) == list(features.columns)
    assert features[builder.numeric_features()].isna().sum().sum() == 0


def test_does_not_use_leakage_or_target(sample: pd.DataFrame) -> None:
    """No leakage, post-booking or target column reaches the features."""
    features = BookingFeatureBuilder().fit(sample).transform(sample)
    forbidden = set(config.LEAKAGE_COLUMNS + config.POST_BOOKING_COLUMNS + [config.TARGET])
    assert not forbidden & set(features.columns)


def test_handles_missing_values_and_raw_codes(booking: dict[str, Any]) -> None:
    """Missing agent/children and the raw meal code are normalized; group size defaults to 1."""
    row = pd.DataFrame([{**booking, "agent": None, "children": None, "meal": "Undefined"}]).drop(
        columns=config.GROUP_SIZE_COLUMN
    )
    features = BookingFeatureBuilder().fit(row).transform(row).iloc[0]
    assert features["agent"] == "none" and features["has_agent"] == 0
    assert features["children"] == 0 and features["meal"] == "SC"
    assert features[config.GROUP_SIZE_COLUMN] == 1
    assert features["arrival_month"] == "3" and features["total_nights"] == 3


def test_is_deterministic_and_keeps_index(sample: pd.DataFrame) -> None:
    """Transforming twice gives the same result and the input index is preserved."""
    subset = sample.iloc[10:20]
    builder = BookingFeatureBuilder().fit(subset)
    first, second = builder.transform(subset), builder.transform(subset)
    pd.testing.assert_frame_equal(first, second)
    assert first.index.equals(subset.index)


def test_optional_features_follow_eda_rules(sample: pd.DataFrame, booking: dict[str, Any]) -> None:
    """Lead time bins, crosses and relative price are added when enabled."""
    builder = BookingFeatureBuilder(lead_time_bins=True, crosses=True, relative_adr=True).fit(sample)
    rows = pd.DataFrame([{**booking, "lead_time": 5}, {**booking, "lead_time": 400, "deposit_type": "Non Refund"}])
    features = builder.transform(rows)
    assert features["lead_time_bucket"].tolist() == ["0-7", ">365"]
    assert features["hotel_x_deposit_type"].tolist() == ["City Hotel|No Deposit", "City Hotel|Non Refund"]
    assert (features["adr_relative"] > 0).all()
    assert list(features.columns) == builder.numeric_features() + builder.categorical_features()


def test_relative_adr_uses_prices_learned_at_fit(sample: pd.DataFrame, booking: dict[str, Any]) -> None:
    """The reference price comes from fit data only, so relative price scales with adr."""
    builder = BookingFeatureBuilder(relative_adr=True).fit(sample)
    cheap, pricey = (pd.DataFrame([{**booking, "adr": adr}]) for adr in (50.0, 200.0))
    ratio = builder.transform(pricey)["adr_relative"].iloc[0] / builder.transform(cheap)["adr_relative"].iloc[0]
    assert np.isclose(ratio, 4.0)
