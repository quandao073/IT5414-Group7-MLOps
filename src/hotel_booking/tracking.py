"""Logging and MLflow helpers shared by training and tuning."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from hotel_booking.model import AveragingClassifier, RecencyWeightedClassifier

PARAM_VALUE_LIMIT = 500


def configure_logging(log_dir: Path | None = None) -> None:
    """Send loguru output to stderr and optionally to a rotating file.

    Args:
        log_dir: Folder for `train.log`; no file is written when `None`.
    """
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level: <7} | {message}")
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(log_dir / "train.log", level="DEBUG", rotation="10 MB", retention=5)


def setup_mlflow(experiment: str) -> None:
    """Point MLflow at `MLFLOW_TRACKING_URI` (default: local `mlruns`) and select the experiment.

    Args:
        experiment: Experiment name, created if missing.
    """
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    logger.info(f"MLflow tracking at {mlflow.get_tracking_uri()}, experiment {experiment}")


def git_commit() -> str:
    """Short hash of the current commit.

    Returns:
        The hash, or `unknown` outside a git checkout.
    """
    try:
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def flatten_params(config: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten a nested config into dotted names with string values that MLflow accepts.

    Args:
        config: Nested config.
        prefix: Prefix for every key.

    Returns:
        Dotted parameter name to value, cut to the MLflow length limit.
    """
    flat: dict[str, str] = {}
    for key, value in config.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_params(value, f"{name}."))
        else:
            flat[name] = str(value)[:PARAM_VALUE_LIMIT]
    return flat


def log_confusion_matrix(y_true: Any, proba: Any, threshold: float, name: str = "confusion_matrix.png") -> None:
    """Log the confusion matrix as a figure and as `tn`, `fp`, `fn`, `tp` metrics.

    Args:
        y_true: True labels.
        proba: Predicted cancellation probabilities.
        threshold: Decision threshold.
        name: File name of the figure under the `plots` artifact folder.
    """
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    mlflow.log_metrics({"tn": float(tn), "fp": float(fp), "fn": float(fn), "tp": float(tp)})
    display = ConfusionMatrixDisplay.from_predictions(y_true, pred, labels=[0, 1], colorbar=False)
    display.ax_.set_title(f"Confusion matrix (threshold {threshold:.3f})")
    mlflow.log_figure(display.figure_, f"plots/{name}")
    plt.close(display.figure_)


def single_importance(model: RecencyWeightedClassifier) -> pd.Series:
    """Normalized importance of each encoded feature in a fitted single model.

    Args:
        model: Fitted model whose pipeline ends in a tree model or a linear model.

    Returns:
        Importance per encoded feature, summing to 1.
    """
    pipeline = model.estimator_
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    inner = pipeline.named_steps["model"]
    values = inner.feature_importances_ if hasattr(inner, "feature_importances_") else np.abs(inner.coef_[0])
    series = pd.Series(np.asarray(values, dtype=float), index=names)
    return series / series.sum()


def feature_importance(model: Any) -> pd.DataFrame | None:
    """Feature importance of a fitted model; ensembles average their members.

    Args:
        model: A fitted `RecencyWeightedClassifier` or `AveragingClassifier`.

    Returns:
        A table with `feature` and `importance`, largest first, or `None` for an unsupported model.
    """
    if isinstance(model, RecencyWeightedClassifier):
        series = single_importance(model)
    elif isinstance(model, AveragingClassifier) and all(
        isinstance(member, RecencyWeightedClassifier) for member in model.estimators_
    ):
        series = pd.concat([single_importance(member) for member in model.estimators_], axis=1).fillna(0).mean(axis=1)
    else:
        logger.warning(f"No feature importance for {type(model).__name__}")
        return None
    return series.sort_values(ascending=False).rename_axis("feature").rename("importance").reset_index()


def log_feature_importance(frame: pd.DataFrame, top: int = 30) -> None:
    """Log the full table as CSV and the `top` features as a bar chart.

    Args:
        frame: Output of `feature_importance`.
        top: Number of features in the chart.
    """
    mlflow.log_text(frame.to_csv(index=False), "plots/feature_importance.csv")
    best = frame.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(best) + 1.5))
    ax.barh(best["feature"], best["importance"])
    ax.set_title(f"Top {len(best)} feature importances")
    fig.tight_layout()
    mlflow.log_figure(fig, "plots/feature_importance.png")
    plt.close(fig)
