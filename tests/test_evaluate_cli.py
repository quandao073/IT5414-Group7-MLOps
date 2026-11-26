import json
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from hotel_booking import evaluate, train
from tests.test_train import make_config

runner = CliRunner()


@pytest.fixture
def saved_model(tmp_path: Path, processed_dir: Path) -> Path:
    """Train a tiny model without tracking and return the models folder."""
    train.run_training(make_config(tmp_path, processed_dir, "train.tracking=false"))
    return tmp_path / "models"


def test_cli_prints_headline_metrics_and_writes_json(saved_model: Path, processed_dir: Path, tmp_path: Path) -> None:
    """The CLI prints ROC-AUC, PR-AUC and F1 and can write every metric to a JSON file."""
    output = tmp_path / "metrics.json"
    result = runner.invoke(
        make_app(),
        ["--model-path", str(saved_model / "model.joblib"), "--processed-dir", str(processed_dir)]
        + ["--card-path", str(saved_model / "model_card.json"), "--output", str(output)],
    )
    assert result.exit_code == 0, result.output
    for name in ("roc_auc", "pr_auc", "f1"):
        assert f"{name}:" in result.output
    assert {"roc_auc", "pr_auc", "f1", "brier"} <= json.loads(output.read_text(encoding="utf-8")).keys()


def make_app() -> typer.Typer:
    """Typer app around `evaluate.main`, as `typer.run` builds it."""
    app = typer.Typer()
    app.command()(evaluate.main)
    return app


def test_explicit_threshold_overrides_the_card(saved_model: Path, processed_dir: Path) -> None:
    """A threshold of 0 predicts every booking as canceled, so recall is 1."""
    metrics = evaluate.evaluate_model(saved_model / "model.joblib", processed_dir, threshold=0.0)
    assert metrics["recall"] == 1.0
