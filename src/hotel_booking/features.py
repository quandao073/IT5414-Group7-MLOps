"""Build model features from raw booking records.

Only the `relative_adr` option learns statistics (median prices) at fit time; every other feature is computed
from each record on its own, so training and serving always transform the data the same way.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from hotel_booking.config import GROUP_SIZE_COLUMN
from hotel_booking.data import MONTH_TO_NUMBER

BASE_NUMERIC_FEATURES = [
    "lead_time",
    "arrival_date_week_number",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "total_nights",
    "adults",
    "children",
    "babies",
    "total_guests",
    "is_family",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "previous_cancel_ratio",
    "days_in_waiting_list",
    "adr",
    "adr_per_guest",
    "total_of_special_requests",
    "has_agent",
    "has_company",
    GROUP_SIZE_COLUMN,
]
BASE_CATEGORICAL_FEATURES = [
    "hotel",
    "meal",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
    "agent",
    "arrival_month",
    "arrival_weekday",
]
PASSTHROUGH_NUMERIC = [
    "lead_time",
    "arrival_date_week_number",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "babies",
    "is_repeated_guest",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "days_in_waiting_list",
    "adr",
    "total_of_special_requests",
]
PASSTHROUGH_CATEGORICAL = [
    "hotel",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
]

# Lead time bin edges (days) from EDA: the cancellation rate rises steadily across these bins.
LEAD_TIME_EDGES = [7, 30, 90, 180, 365]
LEAD_TIME_LABELS = ["0-7", "8-30", "31-90", "91-180", "181-365", ">365"]
# Feature crosses from EDA: Non Refund clusters in Groups/Offline TA, and the two hotels cancel differently.
CROSSED_FEATURES = [("hotel", "deposit_type"), ("market_segment", "deposit_type"), ("hotel", "market_segment")]
# Reference price: median adr of the first key group with at least MIN_PRICE_GROUP rows, else the overall median.
PRICE_GROUP_KEYS = [["hotel", "reserved_room_type", "arrival_month"], ["hotel", "arrival_month"]]
MIN_PRICE_GROUP = 30


def cross_name(left: str, right: str) -> str:
    """Name of the crossed feature built from two categorical features.

    Args:
        left: First feature name.
        right: Second feature name.

    Returns:
        The crossed feature name.
    """
    return f"{left}_x_{right}"


CROSSED_NAMES = [cross_name(left, right) for left, right in CROSSED_FEATURES]


class BookingFeatureBuilder(BaseEstimator, TransformerMixin):
    """Normalize raw values and build features; the output is a DataFrame with a fixed column order.

    Attributes:
        lead_time_bins: Add `lead_time_bucket`, the lead time binned at `LEAD_TIME_EDGES`.
        crosses: Add the crossed categorical features in `CROSSED_FEATURES`.
        relative_adr: Add `adr_relative`, adr divided by the median price of the same hotel, room type and
            arrival month learned at fit time.
    """

    def __init__(self, lead_time_bins: bool = False, crosses: bool = False, relative_adr: bool = False) -> None:
        """Store the feature options.

        Args:
            lead_time_bins: Add the binned lead time.
            crosses: Add the crossed categorical features.
            relative_adr: Add the price relative to the learned median price.
        """
        self.lead_time_bins = lead_time_bins
        self.crosses = crosses
        self.relative_adr = relative_adr

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "BookingFeatureBuilder":
        """Learn the reference prices when `relative_adr` is set.

        Args:
            X: Raw bookings.
            y: Ignored.

        Returns:
            The fitted builder.
        """
        if self.relative_adr:
            base = self._base_features(X)
            self.price_medians_ = []
            for keys in PRICE_GROUP_KEYS:
                stats = base.groupby(keys)["adr"].agg(["median", "size"])
                self.price_medians_.append(stats.loc[stats["size"] >= MIN_PRICE_GROUP, "median"])
            self.price_overall_ = float(base["adr"].median())
        self.feature_names_out_ = np.array(self.numeric_features() + self.categorical_features(), dtype=object)
        return self

    def numeric_features(self) -> list[str]:
        """Names of the numeric output features.

        Returns:
            The numeric feature names, in output order.
        """
        names = list(BASE_NUMERIC_FEATURES)
        if self.relative_adr:
            names.append("adr_relative")
        return names

    def categorical_features(self) -> list[str]:
        """Names of the categorical output features.

        Returns:
            The categorical feature names, in output order.
        """
        names = list(BASE_CATEGORICAL_FEATURES)
        if self.lead_time_bins:
            names.append("lead_time_bucket")
        if self.crosses:
            names += CROSSED_NAMES
        return names

    def _base_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute the features that need no options and no fitted state.

        Args:
            X: Raw bookings.

        Returns:
            The base features with a fresh range index.
        """
        raw = X.reset_index(drop=True)
        out = pd.DataFrame(index=raw.index)

        children = pd.to_numeric(raw["children"], errors="coerce").fillna(0)
        babies = raw["babies"].fillna(0)
        adults = raw["adults"].fillna(0)
        parts = pd.DataFrame(
            {
                "year": raw["arrival_date_year"],
                "month": raw["arrival_date_month"].map(MONTH_TO_NUMBER),
                "day": raw["arrival_date_day_of_month"],
            }
        )
        arrival = pd.to_datetime(parts, errors="coerce")

        for column in PASSTHROUGH_NUMERIC:
            out[column] = pd.to_numeric(raw[column], errors="coerce")
        out["children"] = children
        out["total_nights"] = raw["stays_in_weekend_nights"] + raw["stays_in_week_nights"]
        out["total_guests"] = adults + children + babies
        out["is_family"] = ((children + babies) > 0).astype(int)
        previous_total = raw["previous_cancellations"] + raw["previous_bookings_not_canceled"]
        cancel_ratio = raw["previous_cancellations"] / previous_total.where(previous_total > 0)
        out["previous_cancel_ratio"] = cancel_ratio.fillna(0.0)
        out["adr_per_guest"] = raw["adr"] / out["total_guests"].where(out["total_guests"] > 0)
        out["has_agent"] = raw["agent"].notna().astype(int)
        out["has_company"] = raw["company"].notna().astype(int)
        out[GROUP_SIZE_COLUMN] = raw[GROUP_SIZE_COLUMN] if GROUP_SIZE_COLUMN in raw else 1

        for column in PASSTHROUGH_CATEGORICAL:
            out[column] = raw[column].astype("str")
        # The source paper describes "Undefined" and "SC" both as no meal package.
        out["meal"] = raw["meal"].astype("str").replace({"Undefined": "SC"})
        out["agent"] = raw["agent"].map(lambda value: "none" if pd.isna(value) else str(int(value)))
        out["arrival_month"] = arrival.dt.month.astype("Int64").astype("str")
        out["arrival_weekday"] = arrival.dt.dayofweek.astype("Int64").astype("str")
        return out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Build the features.

        Args:
            X: Raw bookings.

        Returns:
            The features, indexed like `X`.
        """
        out = self._base_features(X)
        if self.lead_time_bins:
            bins = [-np.inf] + LEAD_TIME_EDGES + [np.inf]
            out["lead_time_bucket"] = pd.cut(out["lead_time"], bins, labels=LEAD_TIME_LABELS).astype("str")
        if self.crosses:
            for left, right in CROSSED_FEATURES:
                out[cross_name(left, right)] = out[left] + "|" + out[right]
        if self.relative_adr:
            reference = pd.Series(np.nan, index=out.index)
            for keys, medians in zip(PRICE_GROUP_KEYS, self.price_medians_):
                lookup = medians.reindex(pd.MultiIndex.from_frame(out[keys])).to_numpy()
                reference = reference.fillna(pd.Series(lookup, index=out.index))
            reference = reference.fillna(self.price_overall_)
            out["adr_relative"] = (out["adr"] / reference.where(reference > 0)).fillna(1.0)

        out = out[self.numeric_features() + self.categorical_features()]
        out.index = X.index
        return out

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        """Output feature names, as required by scikit-learn.

        Args:
            input_features: Ignored.

        Returns:
            The output feature names.
        """
        return self.feature_names_out_
