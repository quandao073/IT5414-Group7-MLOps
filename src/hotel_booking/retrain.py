"""Retrain the model and promote it when it beats the champion by the margin set in `configs/promotion/`.

Run `uv run invoke retrain` or `uv run src/hotel_booking/retrain.py promotion.min_delta=0.002`. The Airflow `retrain`
DAG (contract C14) calls the same entry point.
"""

import sys
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from hotel_booking.registry import evaluate_and_promote
from hotel_booking.tracking import configure_logging
from hotel_booking.train import load_config, run_training


def retrain(cfg: DictConfig, train: pd.DataFrame | None = None, test: pd.DataFrame | None = None) -> dict[str, Any]:
    """Train a new model, register it as challenger and promote it if it is better than the champion.

    Tracking and registration are always on here, whatever `train.tracking` and `train.register` say, because the
    comparison reads the metrics of the registered runs. The metric and the minimum improvement come from
    `cfg.promotion`.

    Args:
        cfg: Composed config with `model`, `data`, `train` and `promotion` groups.
        train: Training split; read from `cfg.data.processed_dir` when `None`.
        test: Test split; read from `cfg.data.processed_dir` when `None`.

    Returns:
        The test `metrics`, whether the challenger was `promoted`, and the compared values in `comparison`.
    """
    forced = OmegaConf.merge(cfg, {"train": {"tracking": True, "register": True}})
    assert isinstance(forced, DictConfig)
    metrics = run_training(forced, train, test)
    promoted, comparison = evaluate_and_promote(metric=cfg.promotion.metric, min_delta=cfg.promotion.min_delta)
    logger.info(f"Retrain finished, promoted={promoted}")
    return {"metrics": metrics, "promoted": promoted, "comparison": comparison}


def main() -> None:
    """Command line entry point: `retrain.py [hydra overrides]`."""
    load_dotenv()
    cfg = load_config(overrides=sys.argv[1:])
    configure_logging(Path(cfg.train.log_dir) if cfg.train.log_dir else None)
    retrain(cfg)


if __name__ == "__main__":
    main()
