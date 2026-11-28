from pathlib import Path
from typing import Any

import mlflow
import pytest
from fastapi.testclient import TestClient

from hotel_booking import train
from hotel_booking.api import create_app
from hotel_booking.loader import MlflowSource, ModelLoadError, source_from_env
from hotel_booking.registry import MODEL_NAME, promote_challenger


@pytest.fixture
def registry_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """A SQLite MLflow tracking store and registry for one test."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    return uri


def train_and_register(tmp_path: Path, processed_dir: Path, version: str) -> None:
    """Train and register a model exactly as the training module does; the new version becomes challenger."""
    cfg = train.load_config(
        overrides=[
            "model=logreg",
            f"data.processed_dir={processed_dir}",
            f"train.output_dir={tmp_path / version}",
            f"train.model_version={version}",
            "train.register=true",
        ]
    )
    train.run_training(cfg)


def test_api_serves_the_champion_and_follows_promotions(
    registry_uri: str, tmp_path: Path, processed_dir: Path, booking: dict[str, Any]
) -> None:
    """The API serves the champion with the card threshold; /reload switches only after a promotion."""
    train_and_register(tmp_path, processed_dir, "v1")
    promote_challenger()
    with TestClient(create_app(source=MlflowSource(MODEL_NAME))) as client:
        body = client.post("/predict", json=booking).json()
        assert body["model_version"] == f"{MODEL_NAME}/1"
        assert 0 <= body["probability"] <= 1

        train_and_register(tmp_path, processed_dir, "v2")
        assert client.post("/reload").json()["changed"] is False

        promote_challenger()
        reloaded = client.post("/reload").json()
        assert reloaded["previous_version"] == f"{MODEL_NAME}/1"
        assert reloaded["current_version"] == f"{MODEL_NAME}/2"
        assert client.get("/health").json()["model_version"] == f"{MODEL_NAME}/2"


def test_missing_alias_and_onnx_are_load_errors(registry_uri: str) -> None:
    """Without a champion, or with the ONNX backend, loading fails so the API keeps its previous model."""
    with pytest.raises(ModelLoadError, match="No version"):
        MlflowSource(MODEL_NAME).load()
    with pytest.raises(ModelLoadError, match="MODEL_SOURCE=file"):
        MlflowSource(MODEL_NAME, backend="onnx").load()


def test_source_from_env_selects_the_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """MODEL_SOURCE=mlflow reads the model name and alias from the environment, with the C6 defaults."""
    monkeypatch.setenv("MODEL_SOURCE", "mlflow")
    assert source_from_env() == MlflowSource(MODEL_NAME, "champion", "joblib")
    monkeypatch.setenv("MLFLOW_MODEL_NAME", "other")
    monkeypatch.setenv("MODEL_ALIAS", "challenger")
    assert source_from_env() == MlflowSource("other", "challenger", "joblib")
