> Guidance for autonomous coding agents
> Read this before writing, editing, or executing anything in this repo.

# Relevant commands

* The project uses `uv` for management of virtual environments. This means:
  * To install packages, use `uv add <package-name>`.
  * To run Python scripts, use `uv run <script-name>.py`.
  * To run other commands related to Python, prefix them with `uv run `, e.g., `uv run <command>`.
* The project uses `pytest` for testing. To run tests, use `uv run pytest tests/`.
* The project uses `ruff` for linting and formatting:
    * To format code, use `uv run ruff format .`.
    * To lint code, use `uv run ruff check . --fix`.
* The project uses `invoke` for task management. To see available tasks, use `uv run invoke --list` or refer to the
    `tasks.py` file.
* The project uses `dvc` to version data. The raw dataset is tracked by `data/raw/hotel_bookings.csv.dvc`;
    fetch it with `uv run dvc pull` and never commit files under `data/` to git.
* `dvc.yaml` defines the data pipeline `prepare → validate → drift`; run it with `uv run dvc repro` and share the
    outputs with `uv run dvc push`. Each stage also has an invoke task:
    * `uv run invoke preprocess-data` writes `data/processed`.
    * `uv run invoke validate-data` checks the splits with pandera and writes `reports/data_validation.json`.
    * `uv run invoke drift-report` writes the Evidently reports and `summary.csv/json` into `reports/drift/`.
* `uv run invoke serve-drift [--reload]` serves the drift detection service on port 8001 (needs `DATABASE_URL`);
    the Airflow DAG `airflow/dags/monitor_drift.py` calls it. See contract C14 in `docs/source/contracts.md`.
* The project uses `pre-commit` for managing pre-commit hooks. To run all hooks on all files, use
    `uv run pre-commit run --all-files`. For more information, refer to the `.pre-commit-config.yaml` file.
* The project has a Streamlit frontend. Run it with `uv run invoke serve-frontend` (needs the API from
  `uv run invoke serve-api` and `API_URL` set, default `http://localhost:8000`).
* Serving (FastAPI) commands:
    * `uv run invoke sample-model [--spec configs/model/<name>.yaml]` trains a model on `data/processed` and saves
      `models/model.joblib` with `models/model_card.json`.
    * `uv run invoke serve-api [--reload]` serves the API on port 8000; Swagger UI at `/docs`. After writing a new
      model to `models/`, `POST /reload` makes the running API serve it without a restart.
    * `uv run invoke replay [--months 2017-01,2017-02] [--delay 0.5]` replays the 2017 bookings into the API.
* Contracts between modules (API schema, model artifact, replay data, environment variables, Docker) are in
    `docs/source/contracts.md`; follow them, or change them through a PR that updates that file.
* Profiling commands:
    * `uv run invoke profile --target {preprocess,train} --label <label> [--spec configs/model/<name>.yaml] [--runs N]`
      writes `reports/profile/<stem>.json` and a gitignored `<stem>.prof`; the summaries are committed.
    * `uv run invoke profile-figures` renders the before/after PNGs into `reports/figures/`.
    * `uv run snakeviz reports/profile/<stem>.prof` opens one raw profile in the browser.
    * `reports/profiling.md` records the measured before/after numbers and which optimisations were rejected.

# Code style

* Follow existing code style.
* Keep line length within 120 characters.
* Use f-strings for formatting.
* Use type hints
* Do not add inline comments unless absolutely necessary.

# Documentation

* If the project has a `docs/` folder, update documentation there as needed.
* In this case the project will be using `mkdocs` for documentation. To build the docs locally, use
    `uv run mkdocs serve`
* Use existing docstring style.
* Ensure all functions and classes have docstrings.
* Use Google style for docstrings.
* Update this `AGENTS.md` file if any new tools or commands are added to the project.
