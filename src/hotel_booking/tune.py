"""Tune LightGBM with Optuna; the study is one MLflow run and every trial a nested run.

Run `uv run invoke tune` or `uv run src/hotel_booking/tune.py hpo.n_trials=10`. Trials are scored by expanding-window
cross-validation on the train split only; the test split is never used.
"""

import sys
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import optuna
import pandas as pd
import yaml
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig

from hotel_booking.config import MODEL_INPUT_COLUMNS, TARGET, TIME_COLUMN
from hotel_booking.data import quarterly_folds, read_split
from hotel_booking.evaluate import binary_metrics
from hotel_booking.model import ModelSpec, build_from_spec
from hotel_booking.tracking import configure_logging, flatten_params, git_commit, setup_mlflow
from hotel_booking.train import apply_seed, load_config, seed_everything, to_dict


def suggest(trial: optuna.Trial, space: dict[str, Any]) -> dict[str, Any]:
    """Sample one parameter set from the search space.

    Args:
        trial: The Optuna trial.
        space: Parameter name to `low`, `high` and optional `step` or `log`.

    Returns:
        Parameter name to sampled value; integer bounds give integers.
    """
    params: dict[str, Any] = {}
    for name, bounds in space.items():
        low, high = bounds["low"], bounds["high"]
        options = {key: bounds[key] for key in ("step", "log") if bounds.get(key) is not None}
        if isinstance(low, int) and isinstance(high, int):
            params[name] = trial.suggest_int(name, low, high, **options)
        else:
            params[name] = trial.suggest_float(name, low, high, **options)
    return params


def cross_validate(spec: ModelSpec, train: pd.DataFrame, metric: str, trial: optuna.Trial | None = None) -> float:
    """Mean validation metric over the expanding-window folds of the last train year.

    Args:
        spec: Model spec to fit on every fold.
        train: The training split.
        metric: Metric name from `binary_metrics`.
        trial: When given, intermediate values are reported and weak trials pruned.

    Returns:
        The mean metric over folds that have both classes.

    Raises:
        optuna.TrialPruned: If the pruner stops the trial.
    """
    folds, _ = quarterly_folds(train[TIME_COLUMN], year=int(train[TIME_COLUMN].dt.year.max()))
    scores: list[float] = []
    for step, (fit_idx, valid_idx) in enumerate(folds):
        fit, valid = train.iloc[fit_idx], train.iloc[valid_idx]
        if len(fit) == 0 or valid[TARGET].nunique() < 2:
            continue
        model = build_from_spec(spec).fit(fit[MODEL_INPUT_COLUMNS], fit[TARGET])
        proba = model.predict_proba(valid[MODEL_INPUT_COLUMNS])[:, 1]
        scores.append(binary_metrics(valid[TARGET], proba)[metric])
        if trial is not None:
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
    if not scores:
        raise ValueError("No usable cross-validation fold in the train split.")
    return float(np.mean(scores))


def make_objective(spec: ModelSpec, train: pd.DataFrame, space: dict[str, Any], metric: str) -> Any:
    """Build the Optuna objective; each call runs as a nested MLflow run.

    Args:
        spec: Seeded LightGBM spec whose `params` are overridden by the sampled values.
        train: The training split.
        space: Search space for `suggest`.
        metric: Metric to maximize or minimize.

    Returns:
        The objective function.
    """

    def objective(trial: optuna.Trial) -> float:
        """Cross-validate one sampled parameter set.

        Args:
            trial: The Optuna trial.

        Returns:
            The mean cross-validated metric.
        """
        params = suggest(trial, space)
        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            mlflow.log_params(params)
            try:
                value = cross_validate({**spec, "params": {**spec["params"], **params}}, train, metric, trial)
            except optuna.TrialPruned:
                mlflow.set_tag("pruned", "true")
                raise
            mlflow.log_metric(f"cv_{metric}", value)
            return value

    return objective


def write_best_spec(spec: ModelSpec, best_params: dict[str, Any], path: Path, note: str) -> None:
    """Write the spec with the tuned parameters so `train.py model=<name>` can reuse it.

    Args:
        spec: The base LightGBM spec.
        best_params: Tuned parameters.
        path: Output YAML file.
        note: Comment written at the top of the file.
    """
    tuned = {**spec, "params": {**spec["params"], **best_params}}
    tuned["params"].pop("random_state", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {note}\n{yaml.safe_dump(tuned, sort_keys=False)}", encoding="utf-8")


def tune(cfg: DictConfig, train: pd.DataFrame | None = None) -> optuna.Study:
    """Run the study and log it to MLflow.

    Args:
        cfg: Composed config with `model`, `data`, `train` and `hpo` groups.
        train: Training split; read from `cfg.data.processed_dir` when `None`.

    Returns:
        The finished study.

    Raises:
        ValueError: If the configured model is not LightGBM.
    """
    seed = cfg.train.seed
    seed_everything(seed)
    spec = apply_seed(to_dict(cfg.model), seed)
    if spec.get("name") != "LightGBM":
        raise ValueError(f"Tuning supports LightGBM only, got {spec.get('name')}.")
    train = read_split(Path(cfg.data.processed_dir), "train") if train is None else train
    space = to_dict(cfg.hpo.space)
    objective = make_objective(spec, train, space, cfg.hpo.metric)

    setup_mlflow(cfg.train.experiment)
    with mlflow.start_run(run_name="optuna-lgbm"):
        mlflow.log_params(flatten_params(to_dict(cfg)))
        mlflow.set_tag("git_commit", git_commit())
        study = optuna.create_study(
            direction=cfg.hpo.direction,
            sampler=optuna.samplers.TPESampler(seed=seed),
            pruner=optuna.pruners.MedianPruner(),
        )
        study.optimize(objective, n_trials=cfg.hpo.n_trials, timeout=cfg.hpo.timeout)
        mlflow.log_params({f"best.{key}": value for key, value in study.best_params.items()})
        mlflow.log_metric(f"best_cv_{cfg.hpo.metric}", study.best_value)
        mlflow.log_dict(study.best_params, "best_params.json")
        mlflow.log_text(study.trials_dataframe().to_csv(index=False), "trials.csv")
        output = Path(cfg.hpo.output_spec)
        note = f"Tuned by Optuna ({cfg.hpo.n_trials} trials, cv {cfg.hpo.metric} {study.best_value:.4f}, seed {seed})."
        write_best_spec(spec, study.best_params, output, note)
        mlflow.log_artifact(str(output), "spec")
    logger.info(f"Best cv {cfg.hpo.metric} {study.best_value:.4f} with {study.best_params}; spec in {output}")
    return study


def main() -> None:
    """Command line entry point: `tune.py [hydra overrides]`."""
    load_dotenv()
    cfg = load_config("tune", sys.argv[1:])
    configure_logging(Path(cfg.train.log_dir) if cfg.train.log_dir else None)
    tune(cfg)


if __name__ == "__main__":
    main()
