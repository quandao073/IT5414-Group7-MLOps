from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.linear_model import LogisticRegression

from hotel_booking import registry


@pytest.fixture(autouse=True)
def sqlite_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A SQLite tracking store and registry per test."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    mlflow.set_experiment("registry-test")


def log_version(roc_auc: float) -> str:
    """Log a tiny model with a `roc_auc` metric and return its run id."""
    frame = pd.DataFrame({"x": np.arange(20.0)})
    model = LogisticRegression().fit(frame, np.arange(20) % 2)
    with mlflow.start_run() as run:
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle", input_example=frame.head(3))
    return run.info.run_id


def test_register_sets_challenger_and_promote_sets_champion() -> None:
    """A new version is the challenger; promoting it makes it loadable as champion."""
    version = registry.register_model(log_version(0.8))
    assert MlflowClient().get_model_version_by_alias(registry.MODEL_NAME, "challenger").version == version.version
    promoted = registry.promote_challenger()
    assert promoted.version == version.version
    model = registry.load_model_by_alias("champion")
    assert model.predict_proba(pd.DataFrame({"x": [1.0, 2.0]})).shape == (2, 2)


def test_evaluate_and_promote_only_when_better() -> None:
    """The challenger replaces the champion only with a higher metric."""
    registry.register_model(log_version(0.80))
    assert registry.evaluate_and_promote()[0] is True
    registry.register_model(log_version(0.70))
    promoted, info = registry.evaluate_and_promote()
    assert promoted is False and info["champion_version"] == 1
    assert registry.get_version_by_alias("champion").version == 1
    registry.register_model(log_version(0.90))
    assert registry.evaluate_and_promote(min_delta=0.05)[0] is True
    assert registry.get_version_by_alias("champion").version == 3


def test_promote_without_challenger_fails_and_unknown_alias_does_not_load() -> None:
    """Missing aliases raise clear errors."""
    with pytest.raises(ValueError, match="challenger"):
        registry.promote_challenger()
    registry.register_model(log_version(0.8))
    with pytest.raises(MlflowException):
        registry.load_model_by_alias("nope")
