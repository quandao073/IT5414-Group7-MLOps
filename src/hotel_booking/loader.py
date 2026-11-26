"""Load the served model from its source (contracts C2 and C7 in `docs/source/contracts.md`).

Only the file source exists for now; the MLflow Registry source (contract C6) will implement the same interface.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np
import pandas as pd

from hotel_booking.config import MODEL_INPUT_COLUMNS

MODEL_FILE = "model.joblib"
CARD_FILE = "model_card.json"
SOURCES = ("file",)


class ModelLoadError(RuntimeError):
    """The model could not be loaded; the API keeps serving the previous one."""


@dataclass(frozen=True)
class ServedModel:
    """A fitted model with the parts of its card the API needs."""

    model: Any
    version: str
    threshold: float
    schema: dict[str, Any]
    backend: str = "joblib"

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Cancellation probabilities.

        Args:
            frame: Bookings with the columns in `MODEL_INPUT_COLUMNS`.

        Returns:
            One probability per row.
        """
        return self.model.predict_proba(frame[MODEL_INPUT_COLUMNS])[:, 1]


class ModelSource(Protocol):
    """Where the served model comes from."""

    def describe(self) -> str:
        """Human-readable location of the model, for logs and errors."""
        ...

    def load(self) -> ServedModel:
        """Load the current model.

        Raises:
            ModelLoadError: When the model is missing or invalid.
        """
        ...


def file_sha256(path: Path) -> str:
    """SHA-256 of a file.

    Args:
        path: The file.

    Returns:
        The hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class FileSource:
    """A `model.joblib` and its `model_card.json` on disk, re-read on every load."""

    model_path: Path
    card_path: Path

    def describe(self) -> str:
        """Human-readable location of the model."""
        return f"{self.model_path} + {self.card_path}"

    def load(self) -> ServedModel:
        """Load the model and its card.

        When the card has `artifact_sha256`, the model file must match it, so a model half-way through being
        replaced is never served with the card of another model.

        Returns:
            The model ready to serve.

        Raises:
            ModelLoadError: When a file is missing, the checksum does not match or the card lacks a field.
        """
        for path in (self.model_path, self.card_path):
            if not path.exists():
                raise ModelLoadError(f"{path} not found. Run `uv run invoke sample-model` first.")
        try:
            card = json.loads(self.card_path.read_text(encoding="utf-8"))
            expected = card.get("artifact_sha256")
            if expected and file_sha256(self.model_path) != expected:
                raise ModelLoadError(f"{self.model_path} does not match artifact_sha256 of {self.card_path}.")
            return ServedModel(
                model=joblib.load(self.model_path),
                version=str(card["model_version"]),
                threshold=float(card["decision_threshold"]),
                schema=card["input_schema"],
            )
        except ModelLoadError:
            raise
        except Exception as error:
            raise ModelLoadError(f"Cannot load {self.describe()}: {error!r}") from error


def source_from_env(model_path: Path | None = None, card_path: Path | None = None) -> ModelSource:
    """Build the model source from arguments or environment variables (contract C5).

    Args:
        model_path: Model file; defaults to `MODEL_PATH`, then `models/model.joblib`.
        card_path: Model card; defaults to `MODEL_CARD_PATH`, then `models/model_card.json`.

    Returns:
        The model source selected by `MODEL_SOURCE` (default `file`).

    Raises:
        ValueError: For an unknown `MODEL_SOURCE`.
    """
    kind = os.getenv("MODEL_SOURCE", "file")
    if kind not in SOURCES:
        raise ValueError(f"Unsupported MODEL_SOURCE={kind!r}; choose from {SOURCES}.")
    return FileSource(
        model_path or Path(os.getenv("MODEL_PATH", f"models/{MODEL_FILE}")),
        card_path or Path(os.getenv("MODEL_CARD_PATH", f"models/{CARD_FILE}")),
    )
