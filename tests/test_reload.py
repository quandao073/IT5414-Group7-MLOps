import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from hotel_booking import data, sample_model
from hotel_booking.api import create_app
from hotel_booking.loader import FileSource, ModelLoadError, source_from_env

SPEC = {"name": "Logistic Regression", "params": {"C": 0.1}}


@pytest.fixture(scope="module")
def trained(sample: pd.DataFrame) -> tuple[Any, dict[str, Any]]:
    """A small model and its card, trained once on synthetic bookings."""
    splits = data.time_split(sample)
    return sample_model.train_with_card(SPEC, splits["train"], splits["test"], "LogReg", "v1")


def publish(trained: tuple[Any, dict[str, Any]], folder: Path, version: str) -> None:
    """Save the trained model under a new version, as a retraining job would."""
    model, card = trained
    sample_model.save_artifacts(model, {**card, "model_version": version}, folder)


@pytest.fixture
def folder(trained: tuple[Any, dict[str, Any]], tmp_path: Path) -> Path:
    """A model folder holding version v1."""
    publish(trained, tmp_path, "v1")
    return tmp_path


def client_for(folder: Path, reload_token: str | None = "") -> TestClient:
    """An API serving the model files in `folder`; an empty token leaves /reload open."""
    source = FileSource(folder / sample_model.MODEL_FILE, folder / sample_model.CARD_FILE)
    return TestClient(create_app(source=source, reload_token=reload_token))


@pytest.fixture
def client(folder: Path) -> Iterator[TestClient]:
    """API client serving version v1 with an open /reload."""
    with client_for(folder) as test_client:
        yield test_client


def test_reload_serves_the_new_version_without_restart(
    client: TestClient, folder: Path, trained: tuple[Any, dict[str, Any]], booking: dict[str, Any]
) -> None:
    """After a new model is published, /reload switches every endpoint and the metrics to it."""
    publish(trained, folder, "v2")
    body = client.post("/reload").json()
    assert body["previous_version"] == "v1" and body["current_version"] == "v2" and body["changed"] is True
    assert client.get("/health").json()["model_version"] == "v2"
    assert client.post("/predict", json=booking).json()["model_version"] == "v2"
    metrics = client.get("/metrics").text
    assert 'api_model_info{backend="joblib",model_version="v2"} 1.0' in metrics
    assert 'model_version="v1"' not in metrics
    assert 'api_model_reloads_total{result="success"} 1.0' in metrics


def test_reload_of_the_same_model_reports_no_change(client: TestClient) -> None:
    """Reloading without a new model succeeds and says nothing changed."""
    body = client.post("/reload").json()
    assert body["current_version"] == body["previous_version"] == "v1" and body["changed"] is False


def test_failed_reload_keeps_the_previous_model(client: TestClient, folder: Path, booking: dict[str, Any]) -> None:
    """A missing model file makes /reload fail with 503 while predictions keep using the old model."""
    (folder / sample_model.MODEL_FILE).unlink()
    response = client.post("/reload")
    assert response.status_code == 503 and "previous model kept" in response.json()["detail"]
    assert client.post("/predict", json=booking).json()["model_version"] == "v1"
    assert 'api_model_reloads_total{result="failure"} 1.0' in client.get("/metrics").text


def test_reload_refuses_a_card_of_another_model(
    client: TestClient, folder: Path, trained: tuple[Any, dict[str, Any]]
) -> None:
    """A model file that does not match the card checksum is never served."""
    card_path = folder / sample_model.CARD_FILE
    card = json.loads(card_path.read_text(encoding="utf-8"))
    card_path.write_text(json.dumps({**card, "model_version": "v2", "artifact_sha256": "0" * 64}), encoding="utf-8")
    assert client.post("/reload").status_code == 503
    assert client.get("/health").json()["model_version"] == "v1"


def test_reload_loads_a_first_model(tmp_path: Path, trained: tuple[Any, dict[str, Any]]) -> None:
    """An API started without a model can load one later through /reload."""
    with client_for(tmp_path) as empty:
        assert empty.get("/health").json()["model_loaded"] is False
        publish(trained, tmp_path, "v1")
        body = empty.post("/reload").json()
        assert body["previous_version"] is None and body["current_version"] == "v1"


@pytest.mark.parametrize("header, status", [(None, 401), ("wrong", 401), ("secret", 200)])
def test_reload_token_is_required_when_set(folder: Path, header: str | None, status: int) -> None:
    """With a reload token, /reload needs the matching X-Reload-Token header."""
    headers = {"X-Reload-Token": header} if header else {}
    with client_for(folder, reload_token="secret") as guarded:
        assert guarded.post("/reload", headers=headers).status_code == status


def test_file_source_reports_missing_and_corrupt_files(folder: Path, tmp_path_factory: pytest.TempPathFactory) -> None:
    """Missing files and unreadable cards raise ModelLoadError."""
    empty = tmp_path_factory.mktemp("empty")
    with pytest.raises(ModelLoadError, match="not found"):
        FileSource(empty / "model.joblib", empty / "model_card.json").load()
    (folder / sample_model.CARD_FILE).write_text("{not json", encoding="utf-8")
    with pytest.raises(ModelLoadError, match="Cannot load"):
        FileSource(folder / sample_model.MODEL_FILE, folder / sample_model.CARD_FILE).load()


def test_saved_card_records_the_model_checksum(folder: Path) -> None:
    """save_artifacts writes the model checksum into the card and leaves no temporary files."""
    card = json.loads((folder / sample_model.CARD_FILE).read_text(encoding="utf-8"))
    assert len(card["artifact_sha256"]) == 64
    assert not list(folder.glob("*.tmp"))


def test_source_from_env_rejects_unknown_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown MODEL_SOURCE fails at startup instead of serving nothing."""
    monkeypatch.setenv("MODEL_SOURCE", "s3")
    with pytest.raises(ValueError, match="MODEL_SOURCE"):
        source_from_env()
