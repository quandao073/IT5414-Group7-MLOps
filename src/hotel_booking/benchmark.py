"""Compare the joblib and ONNX backends: latency, throughput and artifact size (contract C8).

Measures the model call only (feature building plus inference), not HTTP; load tests cover the API end to end.
Run with `uv run invoke benchmark` after `invoke sample-model` and `invoke export-onnx`.
"""

import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer

from hotel_booking.config import MODEL_INPUT_COLUMNS
from hotel_booking.data import read_split
from hotel_booking.loader import CARD_FILE, MODEL_FILE, served_from_folder
from hotel_booking.onnx_export import max_abs_difference

BACKENDS = ("joblib", "onnx")


def time_calls(predict: Any, frame: pd.DataFrame, repeats: int, warmup: int = 3) -> np.ndarray:
    """Wall-clock milliseconds of repeated `predict(frame)` calls.

    Args:
        predict: Function taking a frame of raw bookings.
        frame: The input.
        repeats: Number of timed calls.
        warmup: Untimed calls first.

    Returns:
        One duration per timed call, in milliseconds.
    """
    for _ in range(warmup):
        predict(frame)
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        predict(frame)
        durations.append((time.perf_counter() - start) * 1000)
    return np.array(durations)


def summarize(durations: np.ndarray, rows: int) -> dict[str, float]:
    """Latency percentiles and throughput.

    Args:
        durations: Milliseconds per call.
        rows: Rows per call.

    Returns:
        p50, p95 and mean in ms, and rows per second at the median.
    """
    p50 = float(np.percentile(durations, 50))
    return {
        "p50_ms": round(p50, 3),
        "p95_ms": round(float(np.percentile(durations, 95)), 3),
        "mean_ms": round(float(durations.mean()), 3),
        "rows_per_second": round(rows / (p50 / 1000), 1),
    }


def artifact_mb(model_dir: Path, backend: str, card: dict[str, Any]) -> float:
    """Size of the files a backend needs, in MB.

    The ONNX backend still loads `model.joblib` for the Python feature builders, so its size is reported apart.

    Args:
        model_dir: Folder with the artifacts.
        backend: `joblib` or `onnx`.
        card: The model card.

    Returns:
        Size in megabytes (10^6 bytes).
    """
    if backend == "joblib":
        files = [model_dir / MODEL_FILE]
    else:
        files = [model_dir / artifact["file"] for artifact in card["onnx"]["artifacts"]]
    return round(sum(path.stat().st_size for path in files) / 1e6, 3)


def run_benchmark(
    model_dir: Path, frame: pd.DataFrame, single_repeats: int = 300, batch_rows: int = 1000, batch_repeats: int = 30
) -> dict[str, Any]:
    """Benchmark both backends on the same bookings.

    Args:
        model_dir: Folder with `model.joblib`, `model_card.json` and the ONNX graphs.
        frame: Raw bookings to score.
        single_repeats: Timed calls with one booking.
        batch_rows: Bookings per batch call.
        batch_repeats: Timed batch calls.

    Returns:
        The results per backend, the parity check and the environment.
    """
    card = json.loads((model_dir / CARD_FILE).read_text(encoding="utf-8"))
    served = {name: served_from_folder(model_dir / MODEL_FILE, model_dir / CARD_FILE, name) for name in BACKENDS}
    rows = frame[MODEL_INPUT_COLUMNS].reset_index(drop=True)
    batch = rows.head(batch_rows)
    single = rows.head(1)
    results: dict[str, Any] = {}
    for name, model in served.items():
        predict = model.model.predict_proba
        results[name] = {
            "single": summarize(time_calls(predict, single, single_repeats), 1),
            "batch": summarize(time_calls(predict, batch, batch_repeats), len(batch)),
            "artifact_mb": artifact_mb(model_dir, name, card),
        }
    return {
        "model_version": card["model_version"],
        "model_name": card.get("model_name"),
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "rows": {"single": 1, "batch": len(batch), "parity": len(rows)},
        "parity_max_abs_difference": max_abs_difference(served["joblib"].model, served["onnx"].model, rows),
        "backends": results,
        "environment": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "system": platform.system(),
        },
    }


def to_markdown(result: dict[str, Any]) -> str:
    """A table of the benchmark for reports.

    Args:
        result: Output of `run_benchmark`.

    Returns:
        Markdown text.
    """
    lines = [
        f"Model `{result['model_version']}`, parity on {result['rows']['parity']} rows: "
        f"max |p_joblib - p_onnx| = {result['parity_max_abs_difference']:.2e}",
        "",
        "| Backend | 1 row p50 (ms) | 1 row p95 (ms) | "
        f"{result['rows']['batch']} rows p50 (ms) | rows/s (batch) | Artifact (MB) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in result["backends"].items():
        lines.append(
            f"| {name} | {values['single']['p50_ms']} | {values['single']['p95_ms']} | {values['batch']['p50_ms']} "
            f"| {values['batch']['rows_per_second']} | {values['artifact_mb']} |"
        )
    return "\n".join(lines)


def main(
    model_dir: Path = Path("models"),
    processed_dir: Path = Path("data/processed"),
    output: Path = Path("models/benchmark.json"),
) -> None:
    """Benchmark the exported model on the test split and save the results.

    Args:
        model_dir: Folder with the artifacts.
        processed_dir: Folder with `test.csv`.
        output: JSON file for the results.
    """
    result = run_benchmark(model_dir, read_split(processed_dir, "test").head(5000))
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(to_markdown(result))


if __name__ == "__main__":
    typer.run(main)
