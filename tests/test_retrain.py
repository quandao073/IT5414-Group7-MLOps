from pathlib import Path

import mlflow
import pytest

from hotel_booking import registry, retrain, train
from tests.test_train import make_config


@pytest.fixture(autouse=True)
def sqlite_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A SQLite tracking store and registry per test."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)


def test_promotion_threshold_is_in_the_hydra_config() -> None:
    """The metric and the minimum improvement are config values that can be overridden."""
    cfg = train.load_config(overrides=["promotion.min_delta=0.01"])
    assert cfg.promotion.metric == "roc_auc" and cfg.promotion.min_delta == 0.01
    assert train.load_config().promotion.min_delta == 0.0


def test_first_retrain_becomes_champion_even_when_tracking_is_off(tmp_path: Path, processed_dir: Path) -> None:
    """Retraining forces tracking and registration, and with no champion the new model is promoted."""
    cfg = make_config(tmp_path, processed_dir, "train.tracking=false", "train.register=false")
    result = retrain.retrain(cfg)
    assert result["promoted"] is True and result["comparison"]["reason"] == "no champion yet"
    assert registry.get_version_by_alias("champion").version == 1
    assert {"roc_auc", "pr_auc", "f1"} <= result["metrics"].keys()


def test_min_delta_from_config_decides_the_promotion(tmp_path: Path, processed_dir: Path) -> None:
    """A margin no model can reach keeps the champion; a negative margin lets the challenger win."""
    retrain.retrain(make_config(tmp_path, processed_dir))
    strict = retrain.retrain(make_config(tmp_path, processed_dir, "promotion.min_delta=0.5"))
    assert strict["promoted"] is False
    assert registry.get_version_by_alias("champion").version == 1
    assert registry.get_version_by_alias("challenger").version == 2
    lenient = retrain.retrain(make_config(tmp_path, processed_dir, "promotion.min_delta=-1.0"))
    assert lenient["promoted"] is True
    assert registry.get_version_by_alias("champion").version == 3
