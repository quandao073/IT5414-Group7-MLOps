"""Load the served model from its source (contracts C2, C6 and C7 in `docs/source/contracts.md`).

Sources: `file` (a folder on disk) and `mlflow` (an alias in the MLflow Model Registry, written by training).
"""

import hashlib
import io
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import joblib
import numpy as np
import pandas as pd

from hotel_booking.config import MODEL_INPUT_COLUMNS

MODEL_FILE = "model.joblib"
CARD_FILE = "model_card.json"
SOURCES = ("file", "mlflow")
BACKENDS = ("joblib", "onnx")


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


def served_from_folder(model_path: Path, card_path: Path, backend: str) -> ServedModel:
    """Build the served model from a joblib file, its card and, for ONNX, the graphs listed in the card.

    Args:
        model_path: The `model.joblib` file.
        card_path: Its `model_card.json`.
        backend: `joblib` or `onnx`.

    Returns:
        The model ready to serve.

    Raises:
        ModelLoadError: When a file is missing, a checksum does not match or the card lacks a field.
    """
    for path in (model_path, card_path):
        if not path.exists():
            raise ModelLoadError(f"{path} not found. Run `uv run invoke sample-model` first.")
    try:
        card = json.loads(card_path.read_text(encoding="utf-8"))
        expected = card.get("artifact_sha256")
        # Hash and load the same bytes, so a file replaced between the two steps cannot slip past the check.
        data = model_path.read_bytes()
        if expected and hashlib.sha256(data).hexdigest() != expected:
            raise ModelLoadError(f"{model_path} does not match artifact_sha256 of {card_path}.")
        model = joblib.load(io.BytesIO(data))
        if backend == "onnx":
            model = onnx_model(model, card, model_path.parent)
        return ServedModel(
            model=model,
            version=str(card["model_version"]),
            threshold=float(card["decision_threshold"]),
            schema=card["input_schema"],
            backend=backend,
        )
    except ModelLoadError:
        raise
    except Exception as error:
        raise ModelLoadError(f"Cannot load {model_path} + {card_path}: {error!r}") from error


def onnx_model(model: Any, card: dict[str, Any], folder: Path) -> Any:
    """Wrap the joblib model's feature builders around the ONNX graphs listed in its card.

    Args:
        model: The fitted joblib model.
        card: Its model card, with the `onnx` section written by `invoke export-onnx`.
        folder: Folder holding the ONNX files.

    Returns:
        An `onnx_export.OnnxModel`.

    Raises:
        ModelLoadError: When the card has no ONNX export or a graph is missing or does not match its checksum.
    """
    from hotel_booking.onnx_export import OnnxModel

    artifacts = card.get("onnx", {}).get("artifacts")
    if not artifacts:
        raise ModelLoadError("The model card has no ONNX export. Run `uv run invoke export-onnx` first.")
    graphs = []
    for artifact in artifacts:
        path = folder / artifact["file"]
        graph = path.read_bytes() if path.exists() else b""
        if not graph or hashlib.sha256(graph).hexdigest() != artifact["sha256"]:
            raise ModelLoadError(f"{path} is missing or does not match its sha256 in the model card.")
        graphs.append(graph)
    return OnnxModel(model, graphs)


@dataclass(frozen=True)
class FileSource:
    """A `model.joblib` and its `model_card.json` on disk, re-read on every load."""

    model_path: Path
    card_path: Path
    backend: str = "joblib"

    def describe(self) -> str:
        """Human-readable location of the model."""
        return f"{self.model_path} + {self.card_path} ({self.backend})"

    def load(self) -> ServedModel:
        """Load the model and its card.

        When the card has `artifact_sha256`, the model file must match it, so a model half-way through being
        replaced is never served with the card of another model; the ONNX graphs are checked the same way.

        Returns:
            The model ready to serve.

        Raises:
            ModelLoadError: When a file is missing, a checksum does not match or the card lacks a field.
        """
        return served_from_folder(self.model_path, self.card_path, self.backend)


@dataclass(frozen=True)
class MlflowSource:
    """The version behind an alias of a model in the MLflow Model Registry (contract C6), looked up on every load."""

    name: str
    alias: str = "champion"
    backend: str = "joblib"

    def describe(self) -> str:
        """Human-readable location of the model."""
        return f"models:/{self.name}@{self.alias} ({self.backend})"

    def load(self) -> ServedModel:
        """Load the model of the version behind the alias and the `model_card.json` of its training run.

        Returns:
            The model ready to serve; its version is `<name>/<registry version>`.

        Raises:
            ModelLoadError: When the alias is not set, MLflow is unreachable or the run has no model card.
        """
        if self.backend != "joblib":
            # shortcut: a registered version holds no ONNX graphs (C6), so ONNX is served from files only;
            # log the graphs with the run when ONNX must come from the registry.
            raise ModelLoadError("MODEL_BACKEND=onnx works with MODEL_SOURCE=file only.")
        try:
            import mlflow

            from hotel_booking.registry import get_version_by_alias

            version = get_version_by_alias(self.alias, self.name)
            if version is None:
                raise ModelLoadError(f"No version behind {self.describe()}, or MLflow is unreachable.")
            with tempfile.TemporaryDirectory() as tmp:
                card_file = mlflow.artifacts.download_artifacts(
                    run_id=version.run_id, artifact_path=CARD_FILE, dst_path=tmp
                )
                card = json.loads(Path(card_file).read_text(encoding="utf-8"))
            return ServedModel(
                model=mlflow.sklearn.load_model(f"models:/{self.name}/{version.version}"),
                version=f"{self.name}/{version.version}",
                threshold=float(card["decision_threshold"]),
                schema=card["input_schema"],
            )
        except ModelLoadError:
            raise
        except Exception as error:
            raise ModelLoadError(f"Cannot load {self.describe()}: {error!r}") from error


def backend_from_env() -> str:
    """The inference backend from `MODEL_BACKEND` (default `joblib`).

    Raises:
        ValueError: For an unknown backend.
    """
    backend = os.getenv("MODEL_BACKEND", "joblib")
    if backend not in BACKENDS:
        raise ValueError(f"Unsupported MODEL_BACKEND={backend!r}; choose from {BACKENDS}.")
    return backend


def source_from_env(model_path: Path | None = None, card_path: Path | None = None) -> ModelSource:
    """Build the model source from arguments or environment variables (contract C5).

    Args:
        model_path: Model file; defaults to `MODEL_PATH`, then `models/model.joblib`.
        card_path: Model card; defaults to `MODEL_CARD_PATH`, then `models/model_card.json`.

    Returns:
        The model source selected by `MODEL_SOURCE` (default `file`), using `MODEL_BACKEND`. `mlflow` serves the
        `MODEL_ALIAS` (default `champion`) version of `MLFLOW_MODEL_NAME` from `MLFLOW_TRACKING_URI`.

    Raises:
        ValueError: For an unknown `MODEL_SOURCE` or `MODEL_BACKEND`.
    """
    kind = os.getenv("MODEL_SOURCE", "file")
    if kind not in SOURCES:
        raise ValueError(f"Unsupported MODEL_SOURCE={kind!r}; choose from {SOURCES}.")
    if kind == "mlflow":
        from hotel_booking.registry import MODEL_NAME

        return MlflowSource(
            os.getenv("MLFLOW_MODEL_NAME", MODEL_NAME), os.getenv("MODEL_ALIAS", "champion"), backend_from_env()
        )
    return FileSource(
        model_path or Path(os.getenv("MODEL_PATH", f"models/{MODEL_FILE}")),
        card_path or Path(os.getenv("MODEL_CARD_PATH", f"models/{CARD_FILE}")),
        backend_from_env(),
    )
