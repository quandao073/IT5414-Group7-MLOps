"""Tests for the data-changed workflow contract."""

from pathlib import Path
from typing import Any, cast

import yaml

WORKFLOW = Path(".github/workflows/data_changed.yaml")


def load_workflow() -> dict[Any, Any]:
    """Parse the workflow file.

    Returns:
        The parsed workflow; PyYAML maps the `on` key to `True`.
    """
    parsed: dict[Any, Any] = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return parsed


def triggers_of(config: dict[Any, Any]) -> dict[str, Any]:
    """Return the trigger mapping of a parsed workflow.

    Args:
        config: The parsed workflow.

    Returns:
        The value of the `on` key, whichever way PyYAML spelled it.
    """
    triggers = config.get("on") or config.get(True)
    return cast(dict[str, Any], triggers)


def test_workflow_triggers_on_dvc_changes() -> None:
    """The workflow runs on .dvc changes for push and pull_request, plus manual runs."""
    triggers = triggers_of(load_workflow())

    assert "workflow_dispatch" in triggers
    for event in ("push", "pull_request"):
        assert triggers[event]["branches"] == ["main", "develop"]
        assert "**/*.dvc" in triggers[event]["paths"]
        assert "dvc.lock" in triggers[event]["paths"]
        assert "src/hotel_booking/validation.py" in triggers[event]["paths"]


def test_workflow_checks_secrets_before_pulling() -> None:
    """The job fails on missing DagsHub secrets before it pulls and validates."""
    job = load_workflow()["jobs"]["data"]
    script = "\n".join(step.get("run", "") for step in job["steps"])

    assert job["runs-on"] == "ubuntu-latest"
    assert job["timeout-minutes"] == 15
    assert "secrets.DAGSHUB_USERNAME" in script or "DAGSHUB_USERNAME" in script
    assert "DAGSHUB_TOKEN" in script
    assert script.index("::error::") < script.index("dvc pull")
    assert "dvc pull" in script
    assert script.index("dvc pull") < script.index("invoke preprocess-data")
    assert script.index("invoke preprocess-data") < script.index("invoke validate-data")
    assert "pytest tests/test_validation.py tests/test_data.py" in script


def test_workflow_uploads_the_validation_report() -> None:
    """The validation report is uploaded even when validation fails."""
    steps = load_workflow()["jobs"]["data"]["steps"]
    upload = next(step for step in steps if str(step.get("uses", "")).startswith("actions/upload-artifact"))

    assert upload["if"] == "always()"
    assert upload["with"]["path"] == "reports/data_validation.json"
