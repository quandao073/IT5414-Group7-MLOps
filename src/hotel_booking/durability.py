"""Model durability: ROC-AUC over time and across hotels.

The monthly table scores one model fitted on the training period against every 2017 month. The transfer table fits
the model on one hotel and scores it on the other, next to the same model's in-domain score.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from hotel_booking.config import MODEL_INPUT_COLUMNS, RANDOM_STATE, TARGET, TIME_COLUMN
from hotel_booking.evaluate import binary_metrics
from hotel_booking.model import ModelSpec, build_from_spec

MONTH_FORMAT = "%Y-%m"
BOOTSTRAP_ROUNDS = 500
CONFIDENCE = 95.0
HOTELS = ("City Hotel", "Resort Hotel")
HOTEL_COLUMN = "hotel"
METRIC_COLUMNS = ["month", "rows", "cancel_rate", "roc_auc", "ci_low", "ci_high", "average_precision"]
TRANSFER_COLUMNS = [
    "source_hotel",
    "target_hotel",
    "rows",
    "cancel_rate",
    "roc_auc",
    "average_precision",
    "in_domain",
    "transfer_gap",
]


def bootstrap_roc_auc(
    y_true: Any, proba: Any, rounds: int = BOOTSTRAP_ROUNDS, seed: int = RANDOM_STATE
) -> tuple[float, float, float]:
    """ROC-AUC with a percentile bootstrap interval.

    Args:
        y_true: True labels.
        proba: Predicted cancellation probabilities.
        rounds: Number of bootstrap resamples. Zero returns the point estimate as the whole interval.
        seed: Seed of the resampling generator.

    Returns:
        The point estimate, the lower and the upper bound of the interval.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    score = float(roc_auc_score(y_true, proba))
    if rounds == 0:
        return score, score, score
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(rounds):
        rows = rng.integers(0, len(y_true), len(y_true))
        sample = y_true[rows]
        if len(np.unique(sample)) < 2:
            continue
        samples.append(roc_auc_score(sample, proba[rows]))
    if not samples:
        return score, score, score
    tail = (100.0 - CONFIDENCE) / 2.0
    low, high = np.percentile(samples, [tail, 100.0 - tail])
    return score, float(min(low, score)), float(max(high, score))


def monthly_metrics(model: Any, frame: pd.DataFrame) -> pd.DataFrame:
    """Score a fitted model on every calendar month of a frame.

    Months with a single class have no ROC-AUC and are left out.

    Args:
        model: Fitted model with `predict_proba`.
        frame: Labelled rows with the arrival date and the target.

    Returns:
        One row per month, in calendar order.
    """
    months = pd.to_datetime(frame[TIME_COLUMN]).dt.strftime(MONTH_FORMAT)
    rows = []
    for month, group in frame.groupby(months, sort=True):
        y_true = group[TARGET].to_numpy()
        if len(np.unique(y_true)) < 2:
            continue
        proba = model.predict_proba(group)[:, 1]
        roc_auc, ci_low, ci_high = bootstrap_roc_auc(y_true, proba)
        rows.append(
            {
                "month": month,
                "rows": len(group),
                "cancel_rate": float(y_true.mean()),
                "roc_auc": roc_auc,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "average_precision": float(average_precision_score(y_true, proba)),
            }
        )
    return pd.DataFrame(rows, columns=METRIC_COLUMNS)


def hotel_transfer_metrics(train: pd.DataFrame, test: pd.DataFrame, spec: ModelSpec) -> pd.DataFrame:
    """Fit on one hotel and score every hotel, in-domain and transferred.

    A source or target hotel with no rows or a single class is skipped. The gap is the source hotel's in-domain
    ROC-AUC minus the row's ROC-AUC, zero on the diagonal and missing when the diagonal was skipped.

    Args:
        train: Labelled rows used to fit the model.
        test: Labelled rows used to score it.
        spec: Model recipe passed to `build_from_spec`.

    Returns:
        One row per source/target hotel pair.
    """
    rows = []
    diagonal = {}
    for source in HOTELS:
        source_train = train[train[HOTEL_COLUMN] == source]
        if source_train.empty or source_train[TARGET].nunique() < 2:
            continue
        model = build_from_spec(spec)
        model.fit(source_train[MODEL_INPUT_COLUMNS], source_train[TARGET])
        for target in HOTELS:
            target_test = test[test[HOTEL_COLUMN] == target]
            if target_test.empty or target_test[TARGET].nunique() < 2:
                continue
            proba = model.predict_proba(target_test[MODEL_INPUT_COLUMNS])[:, 1]
            metrics = binary_metrics(target_test[TARGET], proba)
            if source == target:
                diagonal[source] = metrics["roc_auc"]
            rows.append(
                {
                    "source_hotel": source,
                    "target_hotel": target,
                    "rows": len(target_test),
                    "cancel_rate": float(target_test[TARGET].mean()),
                    "roc_auc": metrics["roc_auc"],
                    "average_precision": metrics["average_precision"],
                    "in_domain": source == target,
                }
            )
    table = pd.DataFrame(rows, columns=TRANSFER_COLUMNS[:-1])
    gaps = []
    for row in table.itertuples():
        if row.in_domain:
            gaps.append(0.0)
        else:
            gaps.append(diagonal.get(row.source_hotel, float("nan")) - row.roc_auc)
    table["transfer_gap"] = gaps
    return table
