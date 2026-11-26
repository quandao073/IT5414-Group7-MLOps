"""Check data drift on a schedule and trigger the `retrain` DAG when the drift share crosses the threshold.

Like every DAG here it only orchestrates: drift is computed by the drift service (contract C14) over HTTP, so the
Airflow image needs nothing beyond the standard library and the standard provider.

Airflow Variables, editable in the UI during a demo:
- `drift_share_threshold` (default 0.5): retrain when the share of drifted columns reaches it.
- `monitor_drift_last_window`: the last window that triggered a retrain, so one month triggers at most once.

Environment: `DRIFT_SERVICE_URL` (default `http://drift:8001`), `MONITOR_DRIFT_SCHEDULE` (default every 10 minutes).
"""

import json
import os
import urllib.error
import urllib.request
from datetime import datetime
from typing import Any

from airflow.providers.standard.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.sdk import Variable, dag, task

DRIFT_SERVICE_URL = os.getenv("DRIFT_SERVICE_URL", "http://drift:8001")
SCHEDULE = os.getenv("MONITOR_DRIFT_SCHEDULE", "*/10 * * * *")
RETRAIN_DAG_ID = "retrain"
THRESHOLD_VARIABLE = "drift_share_threshold"
LAST_WINDOW_VARIABLE = "monitor_drift_last_window"
DEFAULT_THRESHOLD = 0.5


def should_retrain(result: dict[str, Any], threshold: float, last_window: str | None) -> bool:
    """Decide whether a drift result calls for a retrain.

    Args:
        result: Response of `GET /drift/latest`, or `{"window": None}` when there is nothing to check yet.
        threshold: Drift share at or above which the model is retrained.
        last_window: The window that triggered the previous retrain.

    Returns:
        True when the window is new and its drift share reaches the threshold.
    """
    window = result.get("window")
    return window is not None and window != last_window and float(result["drift_share"]) >= threshold


@dag(
    schedule=SCHEDULE,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["monitoring"],
)
def monitor_drift() -> None:
    """Ask the drift service for the latest replay month and trigger `retrain` when it drifted."""

    @task
    def check_drift() -> dict[str, Any]:
        url = f"{DRIFT_SERVICE_URL}/drift/latest"
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                result = json.loads(response.read().decode())
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
            print(f"Nothing to check yet: {error.read().decode()}")
            return {"window": None, "drift_share": 0.0}
        print(f"{result['window']}: {result['drifted_columns']}/{result['columns']} columns drifted")
        return {key: result[key] for key in ("window", "rows", "drifted_columns", "columns", "drift_share")}

    @task.short_circuit
    def needs_retrain(result: dict[str, Any]) -> bool:
        threshold = float(Variable.get(THRESHOLD_VARIABLE, default=str(DEFAULT_THRESHOLD)))
        last_window = Variable.get(LAST_WINDOW_VARIABLE, default=None)
        decision = should_retrain(result, threshold, last_window)
        print(f"drift_share={result['drift_share']} threshold={threshold} last={last_window} retrain={decision}")
        return decision

    @task
    def remember(result: dict[str, Any]) -> None:
        Variable.set(LAST_WINDOW_VARIABLE, result["window"])

    result = check_drift()
    trigger = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id=RETRAIN_DAG_ID,
        conf={
            "reason": "drift",
            "window": "{{ ti.xcom_pull(task_ids='check_drift')['window'] }}",
            "drift_share": "{{ ti.xcom_pull(task_ids='check_drift')['drift_share'] }}",
        },
    )
    needs_retrain(result) >> trigger >> remember(result)


monitor_drift()
