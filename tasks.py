import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "hotel_booking"
PYTHON_VERSION = "3.12"


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
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"uv run src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)


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
    ctx.run("uv run mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("uv run mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
