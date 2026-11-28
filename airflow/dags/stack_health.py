"""Check that the services of the docker compose stack answer, from inside Airflow.

Triggered by hand from the Airflow UI. It needs nothing beyond the standard library, so it also shows the rule
for every DAG here: Airflow only orchestrates, and project code runs outside the Airflow image.
"""

import json
import urllib.request
from datetime import datetime

from airflow.sdk import dag, task

SERVICES = {
    "api": "http://api:8000/health",
    "mlflow": "http://mlflow:5000/health",
    "drift": "http://drift:8001/health",
}


@dag(schedule=None, start_date=datetime(2026, 1, 1), catchup=False, tags=["platform"])
def stack_health() -> None:
    """Call the health endpoint of each service; the run fails if one does not answer with HTTP 200."""

    @task
    def check(name: str, url: str) -> str:
        with urllib.request.urlopen(url, timeout=10) as response:
            body = response.read().decode()
        print(f"{name}: HTTP {response.status} {body}")
        return json.dumps({"service": name, "status": response.status})

    for name, url in SERVICES.items():
        check.override(task_id=f"check_{name}")(name, url)


stack_health()
