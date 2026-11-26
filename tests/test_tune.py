from pathlib import Path

import mlflow
import optuna
import pandas as pd
import pytest
import yaml
from mlflow import MlflowClient

from hotel_booking import train, tune
from hotel_booking.data import read_split
from hotel_booking.model import build_from_spec


def make_config(tmp_path: Path, processed_dir: Path, *overrides: str):
    """Compose the tuning config with two tiny trials and a SQLite MLflow store."""
    return train.load_config(
        "tune",
        [
            f"data.processed_dir={processed_dir}",
            f"hpo.output_spec={tmp_path / 'lgbm_tuned.yaml'}",
            "hpo.n_trials=3",
            "hpo.space.n_estimators.low=10",
            "hpo.space.n_estimators.high=20",
            "hpo.space.n_estimators.step=10",
            "hpo.space.min_child_samples.low=5",
            "hpo.space.min_child_samples.high=20",
            *overrides,
        ],
    )


@pytest.fixture(autouse=True)
def sqlite_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A SQLite MLflow store per test."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")


def test_suggest_returns_integers_for_integer_bounds() -> None:
    """Integer bounds give integers within bounds; float bounds give floats."""
    trial = optuna.create_study().ask()
    params = tune.suggest(trial, {"a": {"low": 2, "high": 5}, "b": {"low": 0.1, "high": 1.0, "log": True}})
    assert isinstance(params["a"], int) and 2 <= params["a"] <= 5
    assert isinstance(params["b"], float) and 0.1 <= params["b"] <= 1.0


def test_tune_logs_parent_run_with_nested_trials(tmp_path: Path, processed_dir: Path) -> None:
    """Every trial is a child run of the study run and the best spec can be built."""
    cfg = make_config(tmp_path, processed_dir)
    study = tune.tune(cfg)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(cfg.train.experiment)
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    (parent,) = [run for run in runs if run.info.run_name == "optuna-lgbm"]
    children = [run for run in runs if run.data.tags.get("mlflow.parentRunId") == parent.info.run_id]
    assert len(children) == len(study.trials) == 3
    assert all(f"cv_{cfg.hpo.metric}" in run.data.metrics or run.data.tags.get("pruned") for run in children)
    spec = yaml.safe_load((tmp_path / "lgbm_tuned.yaml").read_text(encoding="utf-8"))
    assert build_from_spec(spec) is not None
    assert spec["params"]["n_estimators"] in (10, 20)
    assert mlflow.get_run(parent.info.run_id).data.metrics[f"best_cv_{cfg.hpo.metric}"] == pytest.approx(
        study.best_value
    )


def test_same_seed_gives_same_best_params(tmp_path: Path, processed_dir: Path) -> None:
    """The sampler is seeded, so two studies find the same parameters."""
    train_split: pd.DataFrame = read_split(processed_dir, "train")
    first = tune.tune(make_config(tmp_path, processed_dir), train_split)
    second = tune.tune(make_config(tmp_path, processed_dir), train_split)
    assert first.best_params == second.best_params


def test_tune_rejects_other_models(tmp_path: Path, processed_dir: Path) -> None:
    """Only LightGBM can be tuned."""
    with pytest.raises(ValueError, match="LightGBM"):
        tune.tune(make_config(tmp_path, processed_dir, "model=logreg"))
