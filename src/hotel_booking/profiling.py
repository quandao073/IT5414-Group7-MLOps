"""Profile the preprocessing and training entry points with cProfile and render before/after figures."""

import cProfile
import json
import platform
import pstats
import re
import statistics
import tempfile
import time
from collections.abc import Callable
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer
import yaml

from hotel_booking.config import RAW_FILE
from hotel_booking.data import build_input_schema, clean_bookings, load_raw, preprocess, read_split, time_split
from hotel_booking.model import ModelSpec
from hotel_booking.sample_model import train_with_card

PROFILE_DIR = Path("reports/profile")
FIGURE_DIR = Path("reports/figures")
TARGETS = ("preprocess", "train")
TOP_FUNCTIONS = 10
LIBRARIES = ("pandas", "numpy", "scikit-learn", "lightgbm", "xgboost")
BAR_HEIGHT = 0.4


def environment() -> dict[str, str]:
    """Versions recorded next to every measurement.

    Returns:
        The interpreter and platform versions plus one entry per profiled library.
    """
    libraries = {name: version(name) for name in LIBRARIES}
    return {"python": platform.python_version(), "platform": platform.platform(), **libraries}


def stage_timings(raw_dir: Path, output_dir: Path) -> dict[str, float]:
    """Wall-clock seconds of each preprocessing stage, in pipeline order.

    Args:
        raw_dir: Folder that holds `hotel_bookings.csv`.
        output_dir: Folder that receives the split CSV files.

    Returns:
        Seconds spent in each stage, keyed by stage name.
    """
    started = time.perf_counter()
    raw = load_raw(raw_dir / RAW_FILE)
    reading = time.perf_counter() - started

    started = time.perf_counter()
    clean, _ = clean_bookings(raw)
    cleaning = time.perf_counter() - started

    started = time.perf_counter()
    splits = time_split(clean)
    splitting = time.perf_counter() - started

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for name, frame in splits.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    writing = time.perf_counter() - started

    started = time.perf_counter()
    build_input_schema(splits["train"])
    schema = time.perf_counter() - started

    return {
        "read_csv": reading,
        "clean_bookings": cleaning,
        "time_split": splitting,
        "write_splits": writing,
        "input_schema": schema,
    }


def _summarize(wall_seconds: list[float], stages: dict[str, float], runs: int) -> dict[str, Any]:
    """Build the wall-clock section shared by both measurement helpers.

    Args:
        wall_seconds: Timed repetitions, in seconds.
        stages: Per-stage timings, empty for training.
        runs: Number of repetitions that were requested.

    Returns:
        The wall-clock summary of one target.
    """
    return {
        "runs": runs,
        "wall_seconds": wall_seconds,
        "wall_median": statistics.median(wall_seconds) if wall_seconds else 0.0,
        "stages": stages,
    }


def measure_preprocess(raw_dir: Path, runs: int) -> dict[str, Any]:
    """Run `data.preprocess` `runs` times into temporary folders and summarize the wall clock.

    Args:
        raw_dir: Folder that holds `hotel_bookings.csv`.
        runs: Number of timed repetitions.

    Returns:
        Wall-clock seconds per run, their median, and the per-stage timings.
    """
    wall_seconds: list[float] = []
    with tempfile.TemporaryDirectory() as tmp:
        for _ in range(runs):
            started = time.perf_counter()
            preprocess(raw_dir, Path(tmp))
            wall_seconds.append(time.perf_counter() - started)
    with tempfile.TemporaryDirectory() as tmp:
        stages = stage_timings(raw_dir, Path(tmp))
    return _summarize(wall_seconds, stages, runs)


def _training_inputs(spec_path: Path, processed_dir: Path) -> tuple[ModelSpec, str, pd.DataFrame, pd.DataFrame]:
    """Load the spec and the processed splits shared by every training run.

    Args:
        spec_path: YAML model spec.
        processed_dir: Folder that holds `train.csv` and `test.csv`.

    Returns:
        The spec, its display name, the training split and the test split.
    """
    spec: ModelSpec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    name = spec.get("name") or " + ".join(member["name"] for member in spec["ensemble"])
    return spec, name, read_split(processed_dir, "train"), read_split(processed_dir, "test")


def measure_training(spec_path: Path, processed_dir: Path, runs: int) -> dict[str, Any]:
    """Run `train_with_card` `runs` times from the processed splits; write no model artifact.

    Args:
        spec_path: YAML model spec.
        processed_dir: Folder that holds the processed splits.
        runs: Number of timed repetitions.

    Returns:
        Wall-clock seconds per run, their median, and an empty stage mapping.
    """
    spec, name, train, test = _training_inputs(spec_path, processed_dir)
    wall_seconds: list[float] = []
    for _ in range(runs):
        started = time.perf_counter()
        train_with_card(spec, train, test, name, "profile")
        wall_seconds.append(time.perf_counter() - started)
    return _summarize(wall_seconds, {}, runs)


def top_functions(profile: cProfile.Profile, total: float) -> list[dict[str, Any]]:
    """The slowest functions by cumulative time, without harness frames.

    Args:
        profile: Profile populated by one measured run.
        total: Wall-clock seconds of that run.

    Returns:
        At most `TOP_FUNCTIONS` entries with the function name, cumulative seconds and share.
    """
    ranked = sorted(getattr(pstats.Stats(profile), "stats").items(), key=lambda item: item[1][3], reverse=True)
    results: list[dict[str, Any]] = []
    for (filename, line, function), values in ranked:
        if filename.endswith("profiling.py") or filename == "<string>":
            continue
        if len(results) >= TOP_FUNCTIONS:
            break
        cumtime = float(values[3])
        results.append(
            {
                "name": f"{function} ({Path(filename).name}:{line})",
                "cumtime": cumtime,
                "share": cumtime / total if total > 0 else 0.0,
            }
        )
    return results


def profile_target(
    target: str,
    label: str,
    spec_path: Path,
    runs: int,
    raw_dir: Path = Path("data/raw"),
    processed_dir: Path = Path("data/processed"),
) -> Path:
    """Measure one target, dump its profile and write the JSON summary; return the JSON path.

    Every measured run, including the profiled one, writes into a temporary folder, so the
    repository splits and `input_schema.json` are never modified.

    Args:
        target: One of `TARGETS`.
        label: Label of the measurement, for example `before`.
        spec_path: YAML model spec, used by the `train` target.
        runs: Number of plain timed repetitions.
        raw_dir: Folder that holds `hotel_bookings.csv`.
        processed_dir: Folder that holds the processed splits.

    Returns:
        Path of the written JSON summary.

    Raises:
        ValueError: If `target` is not one of `TARGETS`.
    """
    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}, got {target}")

    stem = f"{target}-{label}" if target == "preprocess" else f"{target}-{spec_path.stem}-{label}"

    with tempfile.TemporaryDirectory() as tmp:
        call: Callable[[], Any]
        if target == "preprocess":
            measured = measure_preprocess(raw_dir, runs)
            call = partial(preprocess, raw_dir, Path(tmp))
        else:
            measured = measure_training(spec_path, processed_dir, runs)
            spec, name, train, test = _training_inputs(spec_path, processed_dir)
            call = partial(train_with_card, spec, train, test, name, "profile")

        profiler = cProfile.Profile()
        started = time.perf_counter()
        profiler.runcall(call)
        profiled_total = time.perf_counter() - started

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    profiler.dump_stats(PROFILE_DIR / f"{stem}.prof")

    summary = {
        "target": target,
        "label": label,
        "spec": None if target == "preprocess" else str(spec_path),
        **measured,
        "profiled_total": profiled_total,
        "functions": top_functions(profiler, profiled_total),
        "environment": environment(),
    }
    json_path = PROFILE_DIR / f"{stem}.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return json_path


def _load_pairs(profile_dir: Path) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Load every complete before/after pair of JSON summaries.

    Args:
        profile_dir: Folder that holds the JSON summaries.

    Returns:
        One `(label, before, after)` triple per complete pair, in file order.
    """
    pairs: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for before_path in sorted(profile_dir.glob("*-before.json")):
        after_path = before_path.with_name(before_path.name.replace("-before.json", "-after.json"))
        if not after_path.exists():
            continue
        label = before_path.name.removesuffix("-before.json")
        pairs.append(
            (
                label,
                json.loads(before_path.read_text(encoding="utf-8")),
                json.loads(after_path.read_text(encoding="utf-8")),
            )
        )
    return pairs


def _speedup(before: dict[str, Any], after: dict[str, Any]) -> float:
    """Median wall clock before divided by after, zero when a side is missing.

    Args:
        before: The `before` summary.
        after: The `after` summary.

    Returns:
        The speedup factor, or 0.0 when it cannot be computed.
    """
    after_median = float(after.get("wall_median") or 0.0)
    if after_median <= 0:
        return 0.0
    return float(before.get("wall_median") or 0.0) / after_median


def _align_functions(
    functions_before: list[dict[str, Any]],
    functions_after: list[dict[str, Any]],
) -> tuple[list[str], list[float], list[float]]:
    """Align two function lists by name so both series share one label order.

    The line number is dropped from the identity, so a function that moved by a few lines between the before
    and after runs still lines up.

    Args:
        functions_before: `functions` section of the before summary.
        functions_after: `functions` section of the after summary.

    Returns:
        The union of function names with the before and after cumulative seconds, missing names as 0.0.
    """
    before = {_identity(entry["name"]): float(entry["cumtime"]) for entry in functions_before}
    after = {_identity(entry["name"]): float(entry["cumtime"]) for entry in functions_after}
    labels = sorted(set(before) | set(after))
    return labels, [before.get(name, 0.0) for name in labels], [after.get(name, 0.0) for name in labels]


def _identity(name: str) -> str:
    """Drop the trailing line number from a function label.

    Args:
        name: Function label shaped like `fit (model.py:226)`.

    Returns:
        The same label without the line number.
    """
    return re.sub(r":\d+\)$", ")", name)


def _grouped_bars(ax: Any, labels: list[str], before: list[float], after: list[float], xlabel: str) -> None:
    """Draw two grouped horizontal bars per label.

    Args:
        ax: The matplotlib axes to draw on.
        labels: One category per row.
        before: Before values, in the order of `labels`.
        after: After values, in the order of `labels`.
        xlabel: Label of the value axis.
    """
    positions = np.arange(len(labels), dtype=float)
    ax.barh(positions - BAR_HEIGHT / 2, before, BAR_HEIGHT, label="before")
    ax.barh(positions + BAR_HEIGHT / 2, after, BAR_HEIGHT, label="after")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()
    ax.legend()


def _render_figure(plt: Any, before: dict[str, Any], after: dict[str, Any], out_path: Path, with_stages: bool) -> Path:
    """Render one before/after figure and return its path.

    Args:
        plt: The imported matplotlib pyplot module.
        before: The `before` summary.
        after: The `after` summary.
        out_path: Where the PNG is written.
        with_stages: Whether to add the per-stage panel on the left.

    Returns:
        The written path.
    """
    panels = 2 if with_stages else 1
    fig, axes = plt.subplots(1, panels, figsize=(6 * panels, 6), squeeze=False)
    if with_stages:
        stages_before = before.get("stages") or {}
        stages_after = after.get("stages") or {}
        stage_labels = sorted(set(stages_before) | set(stages_after))
        _grouped_bars(
            axes[0, 0],
            stage_labels,
            [float(stages_before.get(name, 0.0)) for name in stage_labels],
            [float(stages_after.get(name, 0.0)) for name in stage_labels],
            "seconds",
        )
    labels, cumtime_before, cumtime_after = _align_functions(
        before.get("functions") or [], after.get("functions") or []
    )
    _grouped_bars(axes[0, -1], labels, cumtime_before, cumtime_after, "cumulative seconds")
    fig.suptitle(f"Speedup: {_speedup(before, after):.1f}x")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_figures(profile_dir: Path, figure_dir: Path) -> list[Path]:
    """Render the before/after figures from the JSON summaries; skip pairs that are missing.

    Args:
        profile_dir: Folder that holds the JSON summaries.
        figure_dir: Folder that receives the PNG figures.

    Returns:
        The written figure paths, in write order.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = _load_pairs(profile_dir)
    if not pairs:
        return []

    figure_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    selectors = (
        (lambda label: label.startswith("preprocess"), "profile_preprocess.png", True),
        (lambda label: "logistic_regression" in label, "profile_train.png", False),
        (lambda label: "champion" in label, "profile_champion.png", False),
    )
    for select, filename, with_stages in selectors:
        match = next((pair for pair in pairs if select(pair[0])), None)
        if match is None:
            continue
        written.append(_render_figure(plt, match[1], match[2], figure_dir / filename, with_stages))

    fig, ax = plt.subplots(figsize=(10, 6))
    _grouped_bars(
        ax,
        [label for label, _, _ in pairs],
        [float(before.get("wall_median") or 0.0) for _, before, _ in pairs],
        [float(after.get("wall_median") or 0.0) for _, _, after in pairs],
        "wall-clock seconds",
    )
    for row, (_, before, after) in enumerate(pairs):
        median = max(float(before.get("wall_median") or 0.0), float(after.get("wall_median") or 0.0))
        ax.annotate(
            f"{_speedup(before, after):.2f}x", (median, row), textcoords="offset points", xytext=(4, 0), va="center"
        )
    fig.savefig(figure_dir / "profile_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(figure_dir / "profile_summary.png")
    return written


def main(
    target: str = "preprocess",
    label: str = "before",
    spec: Path = Path("configs/model/logistic_regression.yaml"),
    runs: int = 3,
    figures: bool = False,
) -> None:
    """Entry point behind `uv run invoke profile`.

    Args:
        target: One of `TARGETS`.
        label: Label of the measurement, for example `before`.
        spec: YAML model spec, used by the `train` target.
        runs: Number of plain timed repetitions.
        figures: Render the before/after figures instead of measuring.

    Raises:
        ValueError: If `target` is not one of `TARGETS`.
    """
    if figures:
        for path in render_figures(PROFILE_DIR, FIGURE_DIR):
            print(path)
        return

    print(profile_target(target, label, spec, runs))


if __name__ == "__main__":
    typer.run(main)
