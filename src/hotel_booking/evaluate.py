"""Binary classification metrics and decision threshold selection."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(
    y_true: Any, proba: Any, threshold: float = 0.5, sample_weight: np.ndarray | None = None
) -> dict[str, float]:
    """Ranking, threshold and calibration metrics for one set of predictions.

    Args:
        y_true: True labels.
        proba: Predicted cancellation probabilities.
        threshold: Probability at or above which a booking is predicted as canceled.
        sample_weight: Optional row weights, e.g. `1 / rooms_in_booking` to count each booking once.

    Returns:
        Metric name to value.
    """
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= threshold).astype(int)
    w = {"sample_weight": sample_weight}
    return {
        "roc_auc": float(roc_auc_score(y_true, proba, **w)),
        "average_precision": float(average_precision_score(y_true, proba, **w)),
        "f1": float(f1_score(y_true, pred, **w)),
        "precision": float(precision_score(y_true, pred, zero_division=0, **w)),
        "recall": float(recall_score(y_true, pred, **w)),
        "accuracy": float(accuracy_score(y_true, pred, **w)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred, **w)),
        "brier": float(brier_score_loss(y_true, proba, **w)),
        "log_loss": float(log_loss(y_true, proba, **w)),
    }


def best_f1_threshold(y_true: Any, proba: Any) -> tuple[float, pd.DataFrame]:
    """Threshold that maximizes F1.

    Args:
        y_true: True labels.
        proba: Predicted cancellation probabilities.

    Returns:
        The best threshold and a table of precision, recall and F1 per threshold.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    curve = pd.DataFrame({"threshold": thresholds, "precision": precision[:-1], "recall": recall[:-1], "f1": f1})
    return float(thresholds[int(np.argmax(f1))]), curve


def metrics_by_group(y_true: pd.Series, proba: np.ndarray, groups: pd.Series, threshold: float) -> pd.DataFrame:
    """Metrics per group, e.g. per hotel; groups with a single class are skipped.

    Args:
        y_true: True labels.
        proba: Predicted cancellation probabilities, aligned with `y_true`.
        groups: Group label of each row, aligned with `y_true`.
        threshold: Decision threshold.

    Returns:
        One row per group with its size, cancel rate and metrics.
    """
    rows = []
    for key, index in groups.groupby(groups, observed=True).groups.items():
        position = groups.index.get_indexer(index)
        y = y_true.iloc[position]
        if y.nunique() < 2:
            continue
        rows.append(
            {
                "group": str(key),
                "rows": len(position),
                "cancel_rate": y.mean(),
                **binary_metrics(y, proba[position], threshold),
            }
        )
    return pd.DataFrame(rows)
