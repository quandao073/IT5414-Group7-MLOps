"""Binary classification metrics, decision threshold selection and a CLI to evaluate a saved model."""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import typer
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

from hotel_booking.config import MODEL_INPUT_COLUMNS, TARGET
from hotel_booking.data import read_split

HEADLINE_METRICS = ("roc_auc", "pr_auc", "f1")


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


def evaluate_model(
    model_path: Path = Path("models/model.joblib"),
    processed_dir: Path = Path("data/processed"),
    split: str = "test",
    threshold: float | None = None,
    card_path: Path = Path("models/model_card.json"),
    output: Path | None = None,
) -> dict[str, float]:
    """Evaluate a saved model on a processed split.

    Args:
        model_path: Path to `model.joblib`.
        processed_dir: Folder with the splits written by `invoke preprocess-data`.
        split: One of `train`, `test`, `production_holdout`.
        threshold: Decision threshold; defaults to `decision_threshold` in the model card, then 0.5.
        card_path: Path to `model_card.json`.
        output: Optional JSON file to write the metrics to.

    Returns:
        Metric name to value, with `pr_auc` added as an alias of `average_precision`.
    """
    frame = read_split(processed_dir, split)
    if threshold is None:
        card = json.loads(card_path.read_text(encoding="utf-8")) if card_path.exists() else {}
        threshold = float(card.get("decision_threshold", 0.5))
    proba = joblib.load(model_path).predict_proba(frame[MODEL_INPUT_COLUMNS])[:, 1]
    metrics = binary_metrics(frame[TARGET], proba, threshold)
    metrics["pr_auc"] = metrics["average_precision"]
    if output is not None:
        output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main(
    model_path: Path = Path("models/model.joblib"),
    processed_dir: Path = Path("data/processed"),
    split: str = "test",
    threshold: float | None = None,
    card_path: Path = Path("models/model_card.json"),
    output: Path | None = None,
) -> None:
    """Print ROC-AUC, PR-AUC and F1 of a saved model on a split.

    Args:
        model_path: Path to `model.joblib`.
        processed_dir: Folder with the processed splits.
        split: Split to evaluate on.
        threshold: Decision threshold; defaults to the model card's.
        card_path: Path to `model_card.json`.
        output: Optional JSON file for all metrics.
    """
    metrics = evaluate_model(model_path, processed_dir, split, threshold, card_path, output)
    for name in HEADLINE_METRICS:
        print(f"{name}: {metrics[name]:.4f}")


if __name__ == "__main__":
    typer.run(main)
