"""Train a model from a spec and save it with a model card that the API can serve.

The card follows the format of `reports/it3190_model_card.json`, plus a `model_version` field. Until the
training pipeline registers models in MLflow, this gives the API a real artifact to load.
"""

import json
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import typer
import yaml

from hotel_booking.config import MODEL_INPUT_COLUMNS, TARGET, TIME_COLUMN
from hotel_booking.data import build_input_schema, read_split
from hotel_booking.evaluate import best_f1_threshold, binary_metrics
from hotel_booking.model import ModelSpec, build_from_spec

MODEL_FILE = "model.joblib"
CARD_FILE = "model_card.json"
VALIDATION_MONTHS = 3
LIBRARIES = ("scikit-learn", "pandas", "numpy", "lightgbm", "xgboost")


def split_validation(train: pd.DataFrame, months: int = VALIDATION_MONTHS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the last months of train to choose the decision threshold without touching test.

    Args:
        train: The training split, sorted by arrival date.
        months: Number of trailing months to hold out.

    Returns:
        The fitting part and the validation part.
    """
    cutoff = train[TIME_COLUMN].max() - pd.DateOffset(months=months)
    return train[train[TIME_COLUMN] <= cutoff], train[train[TIME_COLUMN] > cutoff]


def describe_split(frame: pd.DataFrame, file: str) -> dict[str, Any]:
    """Summarize a split for the model card.

    Args:
        frame: The split.
        file: File name of the split.

    Returns:
        Rows, date range and cancel rate.
    """
    return {
        "file": file,
        "rows": len(frame),
        "from": f"{frame[TIME_COLUMN].min():%Y-%m-%d}",
        "to": f"{frame[TIME_COLUMN].max():%Y-%m-%d}",
        "cancel_rate": round(float(frame[TARGET].mean()), 4),
    }


def train_with_card(
    spec: ModelSpec, train: pd.DataFrame, test: pd.DataFrame, model_name: str, model_version: str
) -> tuple[Any, dict[str, Any]]:
    """Fit a model on train and build its card.

    The threshold maximizes F1 on the last months of train for a model fitted on the earlier months; the
    served model is then refitted on the whole train split.

    Args:
        spec: Model spec, as in `configs/model/`.
        train: The training split.
        test: The test split, used only to report metrics.
        model_name: Human-readable model name for the card.
        model_version: Version string the API reports with every prediction.

    Returns:
        The fitted model and its card.
    """
    fit_part, valid_part = split_validation(train)
    probe = build_from_spec(spec).fit(fit_part[MODEL_INPUT_COLUMNS], fit_part[TARGET])
    threshold, _ = best_f1_threshold(valid_part[TARGET], probe.predict_proba(valid_part[MODEL_INPUT_COLUMNS])[:, 1])

    model = build_from_spec(spec).fit(train[MODEL_INPUT_COLUMNS], train[TARGET])
    test_proba = model.predict_proba(test[MODEL_INPUT_COLUMNS])[:, 1]
    card = {
        "model_name": model_name,
        "model_version": model_version,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "artifact": MODEL_FILE,
        "input_columns": MODEL_INPUT_COLUMNS,
        "input_schema": build_input_schema(train),
        "model_spec": spec,
        "decision_threshold": threshold,
        "threshold_selection": f"best F1 on the last {VALIDATION_MONTHS} months of train",
        "test_metrics": binary_metrics(test[TARGET], test_proba, threshold),
        "data": {"train": describe_split(train, "train.csv"), "test": describe_split(test, "test.csv")},
        "environment": {"python": platform.python_version(), **{name: version(name) for name in LIBRARIES}},
    }
    return model, card


def save_artifacts(model: Any, card: dict[str, Any], output_dir: Path) -> None:
    """Write the model and its card.

    Args:
        model: The fitted model.
        card: The model card.
        output_dir: Folder for `model.joblib` and `model_card.json`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / MODEL_FILE)
    (output_dir / CARD_FILE).write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")


def main(
    spec_path: Path = Path("configs/model/logistic_regression.yaml"),
    processed_dir: Path = Path("data/processed"),
    output_dir: Path = Path("models"),
    model_version: str = "",
) -> None:
    """Train the model of a spec on `data/processed` and save it for the API.

    Args:
        spec_path: Model spec file.
        processed_dir: Folder with `train.csv` and `test.csv` from `invoke preprocess-data`.
        output_dir: Folder for the artifacts.
        model_version: Version string; defaults to `<spec name>-<UTC timestamp>`.
    """
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    version_label = model_version or f"{spec_path.stem}-{datetime.now(UTC):%Y%m%d%H%M%S}"
    name = spec.get("name") or " + ".join(member["name"] for member in spec["ensemble"])
    model, card = train_with_card(
        spec, read_split(processed_dir, "train"), read_split(processed_dir, "test"), name, version_label
    )
    save_artifacts(model, card, output_dir)
    print(f"Saved {version_label} to {output_dir} (threshold {card['decision_threshold']:.3f})")
    print(json.dumps(card["test_metrics"], indent=2))


if __name__ == "__main__":
    typer.run(main)
