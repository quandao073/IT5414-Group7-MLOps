"""MLflow Model Registry helpers: register, promote by alias, load by alias (contract C6 in `docs/contracts.md`).

Aliases replace the deprecated registry stages: `challenger` is the latest registered candidate and `champion` is
the version that serves.
"""

from typing import Any

import mlflow
import typer
from dotenv import load_dotenv
from loguru import logger
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

MODEL_NAME = "hotel-booking-cancellation"
CHAMPION = "champion"
CHALLENGER = "challenger"
MODEL_ARTIFACT = "model"
# MLflow error codes meaning "no such alias" and "no such registered model".
MISSING_ALIAS_CODES = {"INVALID_PARAMETER_VALUE", "RESOURCE_DOES_NOT_EXIST"}

app = typer.Typer(help="Manage the registered cancellation model.", no_args_is_help=True)


def register_model(run_id: str, name: str = MODEL_NAME, artifact_path: str = MODEL_ARTIFACT) -> ModelVersion:
    """Register the model logged in a run and mark the new version as challenger.

    Args:
        run_id: MLflow run that logged the model at `artifact_path`.
        name: Registered model name, created if missing.
        artifact_path: Artifact path of the logged model.

    Returns:
        The new model version.
    """
    version = mlflow.register_model(f"runs:/{run_id}/{artifact_path}", name)
    MlflowClient().set_registered_model_alias(name, CHALLENGER, version.version)
    logger.info(f"Registered {name} v{version.version} as {CHALLENGER}")
    return version


def set_alias(alias: str, version: int | str, name: str = MODEL_NAME) -> None:
    """Point an alias at a model version; the alias moves if it already exists.

    Args:
        alias: `champion` or `challenger`.
        version: Model version number.
        name: Registered model name.
    """
    MlflowClient().set_registered_model_alias(name, alias, str(version))
    logger.info(f"{name}@{alias} -> v{version}")


def get_version_by_alias(alias: str, name: str = MODEL_NAME) -> ModelVersion | None:
    """Model version that holds an alias.

    Args:
        alias: `champion` or `challenger`.
        name: Registered model name.

    Returns:
        The version, or `None` when the alias is not set.

    Raises:
        MlflowException: For any other error (auth, network), so that a failed lookup is never read as "no champion".
    """
    try:
        return MlflowClient().get_model_version_by_alias(name, alias)
    except MlflowException as error:
        if error.error_code in MISSING_ALIAS_CODES:
            return None
        raise


def promote_challenger(name: str = MODEL_NAME) -> ModelVersion:
    """Make the current challenger the champion.

    Args:
        name: Registered model name.

    Returns:
        The version that is now champion.

    Raises:
        ValueError: If there is no challenger.
    """
    version = get_version_by_alias(CHALLENGER, name)
    if version is None:
        raise ValueError(f"{name} has no {CHALLENGER} version.")
    set_alias(CHAMPION, version.version, name)
    return version


def evaluate_and_promote(name: str = MODEL_NAME, metric: str = "roc_auc", min_delta: float = 0.0) -> tuple[bool, dict]:
    """Promote the challenger when it beats the champion on a metric logged by training.

    Both versions log the metric from `train.py` on the same test split, so the values are comparable. With no
    champion yet, the challenger is promoted.

    Args:
        name: Registered model name.
        metric: Metric name in the source run of each version.
        min_delta: Minimum improvement over the champion.

    Returns:
        Whether the challenger was promoted and the compared values.

    Raises:
        ValueError: If there is no challenger.
    """
    client = MlflowClient()
    challenger = get_version_by_alias(CHALLENGER, name)
    if challenger is None:
        raise ValueError(f"{name} has no {CHALLENGER} version.")
    champion = get_version_by_alias(CHAMPION, name)
    challenger_value = client.get_run(str(challenger.run_id)).data.metrics[metric]
    info: dict[str, Any] = {"metric": metric, "challenger": challenger_value, "challenger_version": challenger.version}
    if champion is None:
        promoted = True
        info["reason"] = "no champion yet"
    else:
        champion_value = client.get_run(str(champion.run_id)).data.metrics[metric]
        info.update({"champion": champion_value, "champion_version": champion.version})
        promoted = challenger_value > champion_value + min_delta
        info["reason"] = f"{metric} {challenger_value:.4f} vs {champion_value:.4f} (min delta {min_delta})"
    if promoted:
        set_alias(CHAMPION, challenger.version, name)
    logger.info(f"Promote challenger v{challenger.version}: {promoted} ({info['reason']})")
    return promoted, info


def load_model_by_alias(alias: str = CHAMPION, name: str = MODEL_NAME) -> Any:
    """Load a registered model by alias.

    Args:
        alias: `champion` or `challenger`.
        name: Registered model name.

    Returns:
        The fitted model; it takes the raw columns in `config.MODEL_INPUT_COLUMNS`.
    """
    return mlflow.sklearn.load_model(f"models:/{name}@{alias}")


@app.command()
def register(run_id: str, name: str = MODEL_NAME) -> None:
    """Register the model of a training run as challenger.

    Args:
        run_id: MLflow run id.
        name: Registered model name.
    """
    register_model(run_id, name)


@app.command()
def promote(name: str = MODEL_NAME, metric: str = "", min_delta: float = 0.0) -> None:
    """Promote the challenger; with `--metric` only when it beats the champion.

    Args:
        name: Registered model name.
        metric: Metric to compare; empty promotes unconditionally.
        min_delta: Minimum improvement over the champion when comparing.
    """
    if metric:
        evaluate_and_promote(name, metric, min_delta)
    else:
        promote_challenger(name)


@app.command("set-alias")
def set_alias_command(alias: str, version: int, name: str = MODEL_NAME) -> None:
    """Point an alias at a version.

    Args:
        alias: Alias name.
        version: Model version number.
        name: Registered model name.
    """
    set_alias(alias, version, name)


@app.command()
def show(name: str = MODEL_NAME) -> None:
    """Print the versions behind the champion and challenger aliases.

    Args:
        name: Registered model name.
    """
    for alias in (CHAMPION, CHALLENGER):
        version = get_version_by_alias(alias, name)
        print(f"{alias}: {'-' if version is None else f'v{version.version} (run {version.run_id})'}")


if __name__ == "__main__":
    load_dotenv()
    app()
