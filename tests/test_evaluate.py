import numpy as np
import pandas as pd

from hotel_booking import evaluate


def test_binary_metrics_on_perfect_ranking() -> None:
    """A perfect ranking scores 1 on ranking metrics and on F1 at a separating threshold."""
    metrics = evaluate.binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9], threshold=0.5)
    assert metrics["roc_auc"] == 1.0 and metrics["average_precision"] == 1.0 and metrics["f1"] == 1.0


def test_best_f1_threshold_separates_classes() -> None:
    """The best F1 threshold is the lowest score of the positive class when classes are separable."""
    threshold, curve = evaluate.best_f1_threshold([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])
    assert threshold == 0.6
    assert curve["f1"].max() == 1.0


def test_metrics_by_group_skips_single_class_groups() -> None:
    """Groups that contain only one class are skipped."""
    y = pd.Series([0, 1, 0, 1, 1])
    proba = np.array([0.2, 0.8, 0.3, 0.7, 0.9])
    groups = pd.Series(["a", "a", "b", "b", "c"])
    table = evaluate.metrics_by_group(y, proba, groups, threshold=0.5)
    assert table["group"].tolist() == ["a", "b"]
