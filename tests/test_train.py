import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pytest
from mlflow import MlflowClient
from omegaconf import DictConfig

from hotel_booking import config, train
from hotel_booking.data import read_split
from hotel_booking.evaluate import evaluate_model
from hotel_booking.model import build_from_spec
from hotel_booking.registry import MODEL_NAME
from hotel_booking.tracking import feature_importance, flatten_params

SMALL_LGBM = ["model.params.n_estimators=20", "model.params.min_child_samples=10"]


def make_config(tmp_path: Path, processed_dir: Path, *overrides: str) -> DictConfig:
    """Compose the training config with the test folders and a local MLflow store."""
    return train.load_config(
        overrides=[
            f"data.processed_dir={processed_dir}",
            f"train.output_dir={tmp_path / 'models'}",
            *SMALL_LGBM,
            *overrides,
        ]
    )


@pytest.fixture
def local_mlflow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Point MLflow at a SQLite store in `tmp_path`, which also supports the model registry."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_registry_uri(uri)
    return uri


def test_train_saves_model_card_and_headline_metrics(tmp_path: Path, processed_dir: Path) -> None:
    """Training writes the model and card the API loads and reports ROC-AUC, PR-AUC and F1."""
    metrics = train.run_training(make_config(tmp_path, processed_dir, "train.tracking=false"))
    models = tmp_path / "models"
    card = json.loads((models / "model_card.json").read_text(encoding="utf-8"))
    assert (models / "model.joblib").exists()
    assert {"model_version", "decision_threshold", "input_schema"} <= card.keys()
    assert card["input_columns"] == config.MODEL_INPUT_COLUMNS
    assert all(0 <= metrics[key] <= 1 for key in ("roc_auc", "pr_auc", "f1"))


def test_evaluate_matches_the_card(tmp_path: Path, processed_dir: Path) -> None:
    """Evaluating the saved model on the test split reproduces the card's metrics."""
    train.run_training(make_config(tmp_path, processed_dir, "train.tracking=false"))
    models = tmp_path / "models"
    card = json.loads((models / "model_card.json").read_text(encoding="utf-8"))
    metrics = evaluate_model(models / "model.joblib", processed_dir, card_path=models / "model_card.json")
    for key in ("roc_auc", "average_precision", "f1"):
        assert metrics[key] == pytest.approx(card["test_metrics"][key])
    assert metrics["pr_auc"] == metrics["average_precision"]


def test_config_groups_and_overrides() -> None:
    """The model group switches the classifier and train.seed can be overridden."""
    cfg = train.load_config(overrides=["model=logreg", "train.seed=7"])
    assert cfg.model.name == "Logistic Regression" and cfg.train.seed == 7
    assert train.load_config().model.name == "LightGBM"


def test_apply_seed_keeps_input_and_seeds_ensemble_members() -> None:
    """Seeds fill missing random_state values, including in ensembles, without touching the input."""
    spec = {"ensemble": [{"name": "LightGBM", "params": {}}, {"name": "XGBoost", "params": {"random_state": 1}}]}
    seeded = train.apply_seed(spec, 5)
    assert [member["params"]["random_state"] for member in seeded["ensemble"]] == [5, 1]
    assert spec["ensemble"][0]["params"] == {}


def test_same_seed_gives_same_model(tmp_path: Path, processed_dir: Path) -> None:
    """Two runs with the same seed predict identically; a different seed does not."""
    test = read_split(processed_dir, "test")[config.MODEL_INPUT_COLUMNS]

    def predict(seed: int) -> np.ndarray:
        cfg = make_config(
            tmp_path, processed_dir, "train.tracking=false", f"train.seed={seed}", "+model.params.n_jobs=1"
        )
        train.run_training(cfg)
        return joblib.load(tmp_path / "models" / "model.joblib").predict_proba(test)

    np.testing.assert_array_equal(predict(1), predict(1))
    assert not np.allclose(predict(1), predict(2))


def test_tracked_run_logs_metrics_plots_and_model(tmp_path: Path, processed_dir: Path, local_mlflow: str) -> None:
    """A tracked run stores params, metrics, confusion matrix, feature importance and the model."""
    train.run_training(make_config(tmp_path, processed_dir))
    client = MlflowClient()
    experiment = client.get_experiment_by_name("hotel-booking-cancellation")
    assert experiment is not None
    (run,) = client.search_runs([experiment.experiment_id])
    assert {"roc_auc", "pr_auc", "f1", "tp", "decision_threshold"} <= run.data.metrics.keys()
    assert run.data.params["model.name"] == "LightGBM"
    plots = {item.path for item in client.list_artifacts(run.info.run_id, "plots")}
    assert {"plots/confusion_matrix.png", "plots/feature_importance.png", "plots/feature_importance.csv"} <= plots
    assert client.list_artifacts(run.info.run_id, "config")


def test_flatten_params_cuts_long_values() -> None:
    """Nested config becomes dotted names and long values fit the MLflow limit."""
    flat = flatten_params({"a": {"b": 1}, "c": "x" * 900})
    assert flat["a.b"] == "1" and len(flat["c"]) == 500


def test_feature_importance_covers_every_encoded_feature(tmp_path: Path, processed_dir: Path) -> None:
    """Importance is normalized over the encoded features for trees, linear models and ensembles."""
    frame = read_split(processed_dir, "train")
    for spec in (
        {"name": "LightGBM", "params": {"n_estimators": 10, "min_child_samples": 5}},
        {"name": "Logistic Regression", "params": {}},
    ):
        model = build_from_spec(spec).fit(frame[config.MODEL_INPUT_COLUMNS], frame[config.TARGET])
        table = feature_importance(model)
        assert table is not None
        assert len(table) == len(model.estimator_.named_steps["preprocess"].get_feature_names_out())
        assert table["importance"].sum() == pytest.approx(1.0)


def test_register_flag_creates_challenger_version(tmp_path: Path, processed_dir: Path, local_mlflow: str) -> None:
    """With train.register=true the run's model becomes the challenger of the registered model."""
    train.run_training(make_config(tmp_path, processed_dir, "train.register=true"))
    version = MlflowClient().get_model_version_by_alias(MODEL_NAME, "challenger")
    assert int(version.version) == 1
