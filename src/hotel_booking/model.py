"""Full model pipeline: raw booking -> features -> encoding -> classifier, built from a model spec."""

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import has_fit_parameter
from xgboost import XGBClassifier

from hotel_booking.config import GROUP_SIZE_COLUMN, RANDOM_STATE
from hotel_booking.data import arrival_dates, recency_weights
from hotel_booking.features import BookingFeatureBuilder

MIN_CATEGORY_FREQUENCY = 30
# Non-negative counts and amounts with a train skew of about 1 or more: log1p before scaling so linear models
# are not dominated by long tails.
LOG_FEATURES = [
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "total_nights",
    "adults",
    "children",
    "babies",
    "total_guests",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "days_in_waiting_list",
    "adr",
    "adr_per_guest",
    "total_of_special_requests",
    GROUP_SIZE_COLUMN,
]
ENCODINGS = ("onehot",)
SCALED_MODELS = {"Logistic Regression"}
MODEL_NAMES = ("Logistic Regression", "Random Forest", "XGBoost", "LightGBM")

ModelSpec = dict[str, Any]


def log1p_nonnegative(X: np.ndarray) -> np.ndarray:
    """Apply log1p after clipping negative values to zero.

    Args:
        X: Numeric values.

    Returns:
        The transformed values.
    """
    return np.log1p(np.clip(X, 0, None))


def make_preprocessor(
    numeric: list[str], categorical: list[str], scale: bool = False, log_features: list[str] | None = None
) -> ColumnTransformer:
    """Impute, optionally log1p and scale numeric columns, and one-hot encode categorical columns.

    Args:
        numeric: Numeric feature names.
        categorical: Categorical feature names.
        scale: Standardize numeric columns.
        log_features: Numeric columns that get log1p before scaling.

    Returns:
        A column transformer with pandas output.
    """
    log_features = log_features or []

    def numeric_pipeline(log: bool) -> Pipeline:
        steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        if log:
            steps.append(("log1p", FunctionTransformer(log1p_nonnegative, feature_names_out="one-to-one")))
        if scale:
            steps.append(("scaler", StandardScaler()))
        return Pipeline(steps)

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist", min_frequency=MIN_CATEGORY_FREQUENCY, sparse_output=False
                ),
            ),
        ]
    )
    logged = [column for column in numeric if column in log_features]
    plain = [column for column in numeric if column not in log_features]
    transformers: list[tuple[str, Any, list[str]]] = []
    if logged:
        transformers.append(("log", numeric_pipeline(log=True), logged))
    transformers.append(("num", numeric_pipeline(log=False), plain))
    transformers.append(("cat", categorical_pipeline, categorical))
    return ColumnTransformer(transformers, verbose_feature_names_out=False).set_output(transform="pandas")


def make_estimator(name: str, **params: Any) -> BaseEstimator:
    """Create a classifier by name; `params` override the defaults.

    Args:
        name: One of `MODEL_NAMES`.
        **params: Estimator parameters.

    Returns:
        The unfitted classifier.
    """
    n_jobs = params.pop("n_jobs", -1)
    if name == "Logistic Regression":
        return LogisticRegression(**{"max_iter": 3000, "random_state": RANDOM_STATE, **params})
    if name == "Random Forest":
        defaults = {"n_estimators": 300, "min_samples_leaf": 3, "n_jobs": n_jobs, "random_state": RANDOM_STATE}
        return RandomForestClassifier(**{**defaults, **params})
    if name == "LightGBM":
        defaults = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "n_jobs": n_jobs,
            "random_state": RANDOM_STATE,
            "verbose": -1,
        }
        return LGBMClassifier(**{**defaults, **params})
    if name == "XGBoost":
        defaults = {
            "n_estimators": 300,
            "learning_rate": 0.1,
            "max_depth": 6,
            "tree_method": "hist",
            "eval_metric": "logloss",
            "n_jobs": n_jobs,
            "random_state": RANDOM_STATE,
        }
        return XGBClassifier(**{**defaults, **params})
    raise ValueError(f"Unsupported model: {name} (choose from {MODEL_NAMES})")


def make_pipeline(
    name: str,
    log_skewed: bool = False,
    lead_time_bins: bool = False,
    crosses: bool = False,
    relative_adr: bool = False,
    **params: Any,
) -> Pipeline:
    """Pipeline that takes the raw columns in `config.MODEL_INPUT_COLUMNS`.

    Args:
        name: Classifier name, one of `MODEL_NAMES`.
        log_skewed: Apply log1p to the columns in `LOG_FEATURES` before scaling.
        lead_time_bins: Passed to `BookingFeatureBuilder`.
        crosses: Passed to `BookingFeatureBuilder`.
        relative_adr: Passed to `BookingFeatureBuilder`.
        **params: Classifier parameters.

    Returns:
        The unfitted pipeline with steps `features`, `preprocess` and `model`.
    """
    builder = BookingFeatureBuilder(lead_time_bins=lead_time_bins, crosses=crosses, relative_adr=relative_adr)
    preprocessor = make_preprocessor(
        builder.numeric_features(),
        builder.categorical_features(),
        scale=name in SCALED_MODELS,
        log_features=LOG_FEATURES if log_skewed else None,
    )
    return Pipeline(
        [
            ("features", builder),
            ("preprocess", preprocessor),
            ("model", make_estimator(name, **params)),
        ]
    )


class RecencyWeightedClassifier(ClassifierMixin, BaseEstimator):
    """Fit a pipeline with sample weights that favor recent bookings and can count each booking once.

    Weights are normalized to a mean of 1. The pipeline's last step must be named `model`, and the input must
    hold the raw arrival date columns and `rooms_in_booking`.

    Attributes:
        estimator: Pipeline to fit.
        half_life_days: Weights halve every `half_life_days` days before the latest arrival date in the
            training data. `None` disables recency weighting.
        booking_weights: Divide each weight by `rooms_in_booking`, so a multi-room booking weighs as much as a
            single-room one.
    """

    def __init__(self, estimator: Pipeline, half_life_days: float | None = None, booking_weights: bool = False) -> None:
        """Store the pipeline and the weighting options.

        Args:
            estimator: Pipeline to fit.
            half_life_days: Recency half-life in days, or `None`.
            booking_weights: Weight each booking once instead of each room.
        """
        self.estimator = estimator
        self.half_life_days = half_life_days
        self.booking_weights = booking_weights

    def sample_weights(self, X: pd.DataFrame) -> np.ndarray | None:
        """Compute the training weights.

        Args:
            X: Raw bookings.

        Returns:
            Weights with mean 1, or `None` when no weighting is configured.
        """
        recency = self.half_life_days is not None and not pd.isna(self.half_life_days) and self.half_life_days > 0
        if not recency and not self.booking_weights:
            return None
        weights = recency_weights(arrival_dates(X), self.half_life_days) if recency else np.ones(len(X))
        if self.booking_weights:
            weights = weights / X[GROUP_SIZE_COLUMN].to_numpy()
        return weights / weights.mean()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RecencyWeightedClassifier":
        """Fit a clone of the pipeline with the sample weights.

        Args:
            X: Raw bookings.
            y: Cancellation labels.

        Returns:
            The fitted classifier.
        """
        self.estimator_ = clone(self.estimator)
        weights = self.sample_weights(X)
        if weights is None:
            self.estimator_.fit(X, y)
        else:
            inner = self.estimator_.named_steps["model"]
            if not has_fit_parameter(inner, "sample_weight"):
                raise ValueError(f"{type(inner).__name__} does not accept sample_weight.")
            self.estimator_.fit(X, y, model__sample_weight=weights)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Raw bookings.

        Returns:
            Probabilities of shape (n_rows, 2).
        """
        return self.estimator_.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict classes at the default 0.5 threshold.

        Args:
            X: Raw bookings.

        Returns:
            Predicted labels.
        """
        return self.estimator_.predict(X)


def make_model(
    name: str, half_life_days: float | None = None, booking_weights: bool = False, **pipeline_kwargs: Any
) -> RecencyWeightedClassifier:
    """Complete model: takes raw bookings and returns cancellation probabilities.

    Args:
        name: Classifier name, one of `MODEL_NAMES`.
        half_life_days: Recency half-life in days, or `None`.
        booking_weights: Weight each booking once instead of each room.
        **pipeline_kwargs: Feature options of `make_pipeline` and classifier parameters.

    Returns:
        The unfitted model.
    """
    return RecencyWeightedClassifier(make_pipeline(name, **pipeline_kwargs), half_life_days, booking_weights)


class AveragingClassifier(ClassifierMixin, BaseEstimator):
    """Average the probabilities of several configured models (soft voting).

    Attributes:
        estimators: (name, model) pairs.
    """

    def __init__(self, estimators: list[tuple[str, BaseEstimator]]) -> None:
        """Store the member models.

        Args:
            estimators: (name, model) pairs.
        """
        self.estimators = estimators

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "AveragingClassifier":
        """Fit a clone of every member.

        Args:
            X: Raw bookings.
            y: Cancellation labels.

        Returns:
            The fitted ensemble.
        """
        self.estimators_ = [clone(estimator).fit(X, y) for _, estimator in self.estimators]
        self.classes_ = self.estimators_[0].classes_
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Average the members' class probabilities.

        Args:
            X: Raw bookings.

        Returns:
            Probabilities of shape (n_rows, 2).
        """
        return np.mean([estimator.predict_proba(X) for estimator in self.estimators_], axis=0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict classes at the default 0.5 threshold.

        Args:
            X: Raw bookings.

        Returns:
            Predicted labels.
        """
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def build_from_spec(spec: ModelSpec) -> RecencyWeightedClassifier | AveragingClassifier:
    """Rebuild a model from its spec (see `configs/model/`).

    A single model spec has `name`, `encoding`, `half_life_days`, `booking_weights`, `pipeline` (feature
    options) and `params`. An ensemble spec is `{"ensemble": [member specs]}`.

    Args:
        spec: The model spec.

    Returns:
        The unfitted model.
    """
    if spec.get("ensemble"):
        return AveragingClassifier([(member["name"], build_from_spec(member)) for member in spec["ensemble"]])
    encoding = spec.get("encoding", "onehot")
    if encoding not in ENCODINGS:
        raise ValueError(f"Unsupported encoding: {encoding} (choose from {ENCODINGS})")
    return make_model(
        spec["name"],
        spec.get("half_life_days"),
        spec.get("booking_weights", False),
        **spec.get("pipeline", {}),
        **spec.get("params", {}),
    )
