"""The retrain decision of the `monitor_drift` DAG.

Airflow is not a project dependency (it runs in its own image), so the DAG module is imported with stand-ins for
the few Airflow names it uses. Importing the DAG against real Airflow is the DagBag check in CI.
"""

import importlib.util
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

DAG_FILE = Path(__file__).resolve().parents[1] / "airflow" / "dags" / "monitor_drift.py"


class _Chain:
    """Stand-in for a task or operator: accepts `>>` and ignores everything else."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Accept any arguments."""

    def __rshift__(self, other: Any) -> Any:
        """Chain tasks."""
        return other


def _decorator(function: Callable[..., Any] | None = None, **kwargs: Any) -> Any:
    """Stand-in for `@dag` and `@task`: calling the decorated function returns a chainable object."""
    if function is None:
        return _decorator
    return lambda *args, **kw: _Chain()


@pytest.fixture(scope="module")
def dag_module() -> types.ModuleType:
    """The DAG module, imported with Airflow stubbed out."""
    sdk = types.ModuleType("airflow.sdk")
    sdk.dag = _decorator  # type: ignore[attr-defined]
    sdk.task = type("Task", (), {"__call__": staticmethod(_decorator), "short_circuit": staticmethod(_decorator)})()  # type: ignore[attr-defined]
    sdk.Variable = types.SimpleNamespace(get=lambda *a, **k: None, set=lambda *a, **k: None)  # type: ignore[attr-defined]
    trigger = types.ModuleType("airflow.providers.standard.operators.trigger_dagrun")
    trigger.TriggerDagRunOperator = _Chain  # type: ignore[attr-defined]
    stubs = {
        "airflow": types.ModuleType("airflow"),
        "airflow.sdk": sdk,
        "airflow.providers": types.ModuleType("airflow.providers"),
        "airflow.providers.standard": types.ModuleType("airflow.providers.standard"),
        "airflow.providers.standard.operators": types.ModuleType("airflow.providers.standard.operators"),
        "airflow.providers.standard.operators.trigger_dagrun": trigger,
    }
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("monitor_drift", DAG_FILE)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    return module


@pytest.mark.parametrize(
    ("result", "last_window", "expected"),
    [
        ({"window": "2017-03", "drift_share": 0.6}, None, True),
        ({"window": "2017-03", "drift_share": 0.5}, "2017-02", True),
        ({"window": "2017-03", "drift_share": 0.4}, None, False),
        ({"window": "2017-03", "drift_share": 0.9}, "2017-03", False),
        ({"window": None, "drift_share": 0.0}, None, False),
    ],
)
def test_should_retrain(dag_module: types.ModuleType, result: dict, last_window: str | None, expected: bool) -> None:
    """Retrain once per new window whose drift share reaches the threshold."""
    assert dag_module.should_retrain(result, 0.5, last_window) is expected


def test_dag_targets_retrain_and_drift_service(dag_module: types.ModuleType) -> None:
    """The DAG triggers the `retrain` DAG and calls the drift service of the compose stack by default."""
    assert dag_module.RETRAIN_DAG_ID == "retrain"
    assert dag_module.DRIFT_SERVICE_URL.endswith(":8001")
