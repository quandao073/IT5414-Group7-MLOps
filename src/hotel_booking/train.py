"""Train the cancellation model from a Hydra config, log the run to MLflow and save `model.joblib` for the API.

Run `uv run invoke train` or `uv run src/hotel_booking/train.py model=logreg train.seed=7`; the config groups are
in `configs/` (override the folder with the `CONFIG_DIR` environment variable).
"""

import copy
import os
import random
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from hotel_booking.config import MODEL_INPUT_COLUMNS, TARGET
from hotel_booking.data import read_split
from hotel_booking.evaluate import HEADLINE_METRICS
from hotel_booking.model import ModelSpec
from hotel_booking.registry import MODEL_ARTIFACT, register_model
from hotel_booking.sample_model import CARD_FILE, save_artifacts, train_with_card
from hotel_booking.tracking import (
    configure_logging,
    feature_importance,
    flatten_params,
    git_commit,
    log_confusion_matrix,
    log_feature_importance,
    setup_mlflow,
)

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_config(config_name: str = "config", overrides: list[str] | None = None) -> DictConfig:
    """Compose a Hydra config from `configs/` without changing the working directory.

    Args:
        config_name: Root config file, e.g. `config` or `tune`.
        overrides: Hydra overrides such as `model=logreg`.

    Returns:
        The composed config.
    """
    config_dir = Path(os.getenv("CONFIG_DIR", DEFAULT_CONFIG_DIR)).resolve()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        return compose(config_name=config_name, overrides=overrides or [])


def to_dict(node: DictConfig) -> dict[str, Any]:
    """Convert a config node to a plain dict with interpolations resolved.

    Args:
        node: A config node.

    Returns:
        The nested dict.
    """
    return cast(dict[str, Any], OmegaConf.to_container(node, resolve=True))


def seed_everything(seed: int) -> None:
    """Seed Python and NumPy so a run is repeatable.

    Args:
        seed: The seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def apply_seed(spec: ModelSpec, seed: int) -> ModelSpec:
    """Copy a model spec and set `random_state` on every model that does not set one.

    Args:
        spec: Single model or ensemble spec.
        seed: The seed.

    Returns:
        The seeded copy; the input is not modified.
    """
    seeded = copy.deepcopy(spec)
    if seeded.get("ensemble"):
        seeded["ensemble"] = [apply_seed(member, seed) for member in seeded["ensemble"]]
    else:
        seeded.setdefault("params", {}).setdefault("random_state", seed)
    return seeded


def spec_name(spec: ModelSpec) -> str:
    """Human-readable model name of a spec.

    Args:
        spec: Single model or ensemble spec.

    Returns:
        The model name, or the member names joined by `+`.
    """
    return spec.get("name") or " + ".join(member["name"] for member in spec["ensemble"])


def default_version(name: str) -> str:
    """Version label `<model name>-<UTC timestamp>`.

    Args:
        name: Human-readable model name.

    Returns:
        The label, e.g. `lightgbm-20261123101500`.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return f"{slug}-{datetime.now(UTC):%Y%m%d%H%M%S}"


def headline(metrics: dict[str, float]) -> dict[str, float]:
    """The ROC-AUC, PR-AUC and F1 of a metrics dict, rounded for logging.

    Args:
        metrics: Output of `binary_metrics` plus `pr_auc`.

    Returns:
        The three headline metrics.
    """
    return {key: round(metrics[key], 4) for key in HEADLINE_METRICS}


def fit_and_save(
    spec: ModelSpec, train: pd.DataFrame, test: pd.DataFrame, version: str, output_dir: Path
) -> tuple[Any, dict[str, Any]]:
    """Train, choose the threshold, save `model.joblib` and `model_card.json`.

    Args:
        spec: Seeded model spec.
        train: The training split.
        test: The test split.
        version: Model version label.
        output_dir: Folder for the artifacts.

    Returns:
        The fitted model and its card; `test_metrics` includes `pr_auc`.
    """
    model, card = train_with_card(spec, train, test, spec_name(spec), version)
    card["test_metrics"]["pr_auc"] = card["test_metrics"]["average_precision"]
    save_artifacts(model, card, output_dir)
    logger.info(f"Saved {version} to {output_dir} (threshold {card['decision_threshold']:.3f})")
    return model, card


def log_run(model: Any, card: dict[str, Any], test: pd.DataFrame, train: pd.DataFrame, output_dir: Path) -> None:
    """Log the metrics, plots and artifacts of a fitted model to the active MLflow run.

    Args:
        model: The fitted model.
        card: Its model card.
        test: The test split.
        train: The training split, used for the model input example.
        output_dir: Folder where the model card was saved.
    """
    threshold = card["decision_threshold"]
    mlflow.log_metrics(card["test_metrics"])
    mlflow.log_metric("decision_threshold", threshold)
    log_confusion_matrix(test[TARGET], model.predict_proba(test[MODEL_INPUT_COLUMNS])[:, 1], threshold)
    importance = feature_importance(model)
    if importance is not None:
        log_feature_importance(importance)
    mlflow.log_artifact(str(output_dir / CARD_FILE))
    mlflow.sklearn.log_model(
        model,
        name=MODEL_ARTIFACT,
        serialization_format="cloudpickle",
        input_example=train[MODEL_INPUT_COLUMNS].head(5),
    )


def run_training(
    cfg: DictConfig, train: pd.DataFrame | None = None, test: pd.DataFrame | None = None
) -> dict[str, float]:
    """Train the configured model, optionally track it in MLflow and register it.

    Args:
        cfg: Composed config with `model`, `data` and `train` groups.
        train: Training split; read from `cfg.data.processed_dir` when `None`.
        test: Test split; read from `cfg.data.processed_dir` when `None`.

    Returns:
        The test metrics, including `pr_auc`.
    """
    seed_everything(cfg.train.seed)
    spec = apply_seed(to_dict(cfg.model), cfg.train.seed)
    processed = Path(cfg.data.processed_dir)
    train = read_split(processed, "train") if train is None else train
    test = read_split(processed, "test") if test is None else test
    version = cfg.train.model_version or default_version(spec_name(spec))
    output_dir = Path(cfg.train.output_dir)
    logger.info(f"Training {spec_name(spec)} {version} on {len(train)} rows, testing on {len(test)} rows")
    logger.debug(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    if not cfg.train.tracking:
        metrics = fit_and_save(spec, train, test, version, output_dir)[1]["test_metrics"]
        logger.info(f"Test metrics: {headline(metrics)}")
        return metrics

    setup_mlflow(cfg.train.experiment)
    with mlflow.start_run(run_name=version) as run:
        mlflow.log_params(flatten_params(to_dict(cfg)))
        mlflow.set_tags({"model_name": spec_name(spec), "model_version": version, "git_commit": git_commit()})
        mlflow.log_text(OmegaConf.to_yaml(cfg), "config/config.yaml")
        model, card = fit_and_save(spec, train, test, version, output_dir)
        log_run(model, card, test, train, output_dir)
        if cfg.train.register:
            registered = register_model(run.info.run_id)
            mlflow.set_tag("registered_version", registered.version)
        logger.info(f"MLflow run {run.info.run_id}: {headline(card['test_metrics'])}")
    return card["test_metrics"]


def main() -> None:
    """Command line entry point: `train.py [hydra overrides]`."""
    load_dotenv()
    cfg = load_config(overrides=sys.argv[1:])
    configure_logging(Path(cfg.train.log_dir) if cfg.train.log_dir else None)
    run_training(cfg)


if __name__ == "__main__":
    main()
