import os
import re
import time
from pathlib import Path

import httpx
from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "hotel_booking"
PYTHON_VERSION = "3.12"


def parse_backend(metrics_text: str) -> str | None:
    """Return the backend label of the api_model_info metric, if present."""
    match = re.search(r'api_model_info\{[^}]*backend="([^"]+)"', metrics_text)
    return match.group(1) if match else None


def wait_for_api(host: str, timeout: float = 60.0, interval: float = 1.0) -> dict[str, object]:
    """Poll GET /health until the API reports status ok and a loaded model."""
    deadline = time.monotonic() + timeout
    last_error = "no response"
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{host}/health", timeout=5.0)
            payload = response.json()
            if response.status_code == 200 and payload.get("status") == "ok" and payload.get("model_loaded") is True:
                return payload
            last_error = f"HTTP {response.status_code}: {payload}"
        except (httpx.HTTPError, ValueError) as error:
            last_error = str(error)
        time.sleep(interval)
    raise RuntimeError(f"API at {host} is not ready after {timeout:.0f}s: {last_error}")


def verify_backend(host: str, backend: str) -> str:
    """Fail unless GET /metrics reports that the API loaded the requested backend."""
    response = httpx.get(f"{host}/metrics", timeout=5.0)
    response.raise_for_status()
    served = parse_backend(response.text)
    if served != backend:
        raise RuntimeError(f"API at {host} serves {served or 'an unknown backend'}, expected {backend}")
    return served


# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"uv run src/{PROJECT_NAME}/data.py data/raw data/processed", echo=True, pty=not WINDOWS)


@task
def validate_data(ctx: Context) -> None:
    """Validate the processed splits with pandera and write reports/data_validation.json."""
    ctx.run(
        f"uv run src/{PROJECT_NAME}/validation.py data/processed reports/data_validation.json",
        echo=True,
        pty=not WINDOWS,
    )


@task
def drift_report(ctx: Context) -> None:
    """Write the Evidently drift reports (each 2017 month and City vs Resort Hotel) into reports/drift."""
    ctx.run(f"uv run src/{PROJECT_NAME}/drift.py data/processed reports/drift", echo=True, pty=not WINDOWS)


@task
def train(ctx: Context, overrides: str = "") -> None:
    """Train with Hydra and track in MLflow; overrides such as "model=logreg train.seed=7 train.register=true"."""
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py {overrides}", echo=True, pty=not WINDOWS)


@task
def evaluate(ctx: Context, split: str = "test") -> None:
    """Print ROC-AUC, PR-AUC and F1 of models/model.joblib on a split."""
    ctx.run(f"uv run src/{PROJECT_NAME}/evaluate.py --split {split}", echo=True, pty=not WINDOWS)


@task
def tune(ctx: Context, overrides: str = "") -> None:
    """Tune LightGBM with Optuna (nested MLflow runs); overrides such as "hpo.n_trials=10"."""
    ctx.run(f"uv run src/{PROJECT_NAME}/tune.py {overrides}", echo=True, pty=not WINDOWS)


@task
def retrain(ctx: Context, overrides: str = "") -> None:
    """Retrain, register as challenger and promote if better; overrides such as "promotion.min_delta=0.002"."""
    ctx.run(f"uv run src/{PROJECT_NAME}/retrain.py {overrides}", echo=True, pty=not WINDOWS)


@task
def registry(ctx: Context, command: str = "show") -> None:
    """Manage the MLflow Model Registry, e.g. "show", "promote --metric roc_auc", "set-alias champion 2"."""
    ctx.run(f"uv run src/{PROJECT_NAME}/registry.py {command}", echo=True, pty=not WINDOWS)


@task
def durability_report(ctx: Context) -> None:
    """Score the champion per 2017 month and across hotels, writing figures into reports/figures."""
    ctx.run(f"uv run src/{PROJECT_NAME}/durability.py", echo=True, pty=not WINDOWS)


@task
def sample_model(ctx: Context, spec: str = "configs/model/logistic_regression.yaml") -> None:
    """Train a model from a spec and save it with its card to models/ for the API."""
    ctx.run(f"uv run src/{PROJECT_NAME}/sample_model.py --spec-path {spec}", echo=True, pty=not WINDOWS)


@task
def export_onnx(ctx: Context) -> None:
    """Export models/model.joblib to ONNX, check it against joblib and record it in the model card."""
    ctx.run(f"uv run src/{PROJECT_NAME}/onnx_export.py", echo=True, pty=not WINDOWS)


@task
def benchmark(ctx: Context) -> None:
    """Compare joblib and ONNX latency, throughput and artifact size on the test split."""
    ctx.run(f"uv run src/{PROJECT_NAME}/benchmark.py", echo=True, pty=not WINDOWS)


@task
def load_test(
    ctx: Context,
    users: int = 20,
    spawn_rate: int = 5,
    run_time: str = "60s",
    host: str = "http://localhost:8000",
    csv_prefix: str = "reports/load_test/run",
    backend: str | None = None,
) -> None:
    """Load test the prediction API with Locust; results go to --csv-prefix."""
    Path(csv_prefix).parent.mkdir(parents=True, exist_ok=True)
    wait_for_api(host)
    if backend is not None:
        print(f"API backend verified: {verify_backend(host, backend)}")
    ctx.run(
        f"uv run locust -f tests/performance/locustfile.py --headless -u {users} -r {spawn_rate} "
        f"-t {run_time} --host {host} --csv {csv_prefix}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def serve_api(ctx: Context, port: int = 8000, reload: bool = False) -> None:
    """Serve the prediction API locally; Swagger UI at /docs."""
    flags = " --reload" if reload else ""
    ctx.run(f"uv run uvicorn {PROJECT_NAME}.api:app --port {port}{flags}", echo=True, pty=not WINDOWS)


@task
def serve_drift(ctx: Context, port: int = 8001, reload: bool = False) -> None:
    """Serve the drift detection service locally (needs DATABASE_URL); Swagger UI at /docs."""
    flags = " --reload" if reload else ""
    ctx.run(f"uv run uvicorn {PROJECT_NAME}.drift_service:app --port {port}{flags}", echo=True, pty=not WINDOWS)


@task
def serve_frontend(ctx: Context, port: int = 8501) -> None:
    """Serve the Streamlit frontend locally."""
    ctx.run(f"uv run streamlit run src/{PROJECT_NAME}/frontend.py --server.port {port}", echo=True, pty=not WINDOWS)


@task
def replay(ctx: Context, api_url: str = "http://localhost:8000", months: str = "", delay: float = 0.0) -> None:
    """Replay the 2017 bookings month by month into the API and release each month's labels."""
    ctx.run(
        f'uv run src/{PROJECT_NAME}/replay.py --api-url {api_url} --months "{months}" --delay {delay}',
        echo=True,
        pty=not WINDOWS,
    )


@task
def profile(
    ctx: Context,
    target: str = "preprocess",
    label: str = "before",
    spec: str = "configs/model/logistic_regression.yaml",
    runs: int = 3,
) -> None:
    """Profile preprocessing or training and write a JSON summary under reports/profile.

    The file name is `preprocess-<label>.json` for preprocessing and `train-<spec>-<label>.json` for training.
    """
    ctx.run(
        f"uv run src/{PROJECT_NAME}/profiling.py --target {target} --label {label} --spec {spec} --runs {runs}",
        echo=True,
        pty=not WINDOWS,
    )


@task
def profile_figures(ctx: Context) -> None:
    """Render the before/after profiling figures into reports/figures."""
    ctx.run(f"uv run src/{PROJECT_NAME}/profiling.py --figures", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}", echo=True, pty=not WINDOWS
    )
    ctx.run(
        f"docker build -t drift:latest . -f dockerfiles/drift.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS,
    )


# Documentation commands
@task
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("uv run mkdocs build --strict --site-dir docs/site", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve", echo=True, pty=not WINDOWS)
