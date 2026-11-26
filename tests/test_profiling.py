"""Tests for the profiling harness."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from hotel_booking import data
from hotel_booking.config import RAW_FILE
from hotel_booking.profiling import (
    _align_functions,
    measure_training,
    profile_target,
    render_figures,
    stage_timings,
)

FUNCTIONS = [
    {"name": "fit (model.py:10)", "cumtime": 1.0, "share": 0.5},
    {"name": "transform (features.py:20)", "cumtime": 0.5, "share": 0.25},
]
OTHER_FUNCTIONS = [{"name": "read_csv (data.py:30)", "cumtime": 0.2, "share": 0.1}]


def write_raw(raw_bookings: pd.DataFrame, tmp_path: Path) -> Path:
    """Write the synthetic bookings into `tmp_path/raw` and return that folder."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / RAW_FILE).write_text(raw_bookings.to_csv(index=False), encoding="utf-8")
    return raw_dir


def summary(
    label: str, target: str, wall_median: float, functions: list[dict[str, Any]], stages: dict[str, float]
) -> dict[str, Any]:
    """Build a summary shaped like the harness output."""
    return {
        "target": target,
        "label": label,
        "spec": None if target == "preprocess" else "configs/model/logistic_regression.yaml",
        "runs": 1,
        "wall_seconds": [wall_median],
        "wall_median": wall_median,
        "stages": stages,
        "profiled_total": wall_median,
        "functions": functions,
        "environment": {"python": "3.12.0", "platform": "test"},
    }


def write_pair(
    profile_dir: Path,
    stem: str,
    target: str,
    before_functions: list[dict[str, Any]],
    after_functions: list[dict[str, Any]],
    stages: dict[str, float] | None = None,
) -> None:
    """Write a complete before/after summary pair."""
    profile_dir.mkdir(parents=True, exist_ok=True)
    for label, wall_median, functions in (
        ("before", 2.0, before_functions),
        ("after", 1.0, after_functions),
    ):
        payload = summary(label, target, wall_median, functions, dict(stages or {}))
        (profile_dir / f"{stem}-{label}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_stage_timings_cover_pipeline(raw_bookings: pd.DataFrame, tmp_path: Path) -> None:
    """`stage_timings` returns the five stages with non-negative seconds."""
    raw_dir = write_raw(raw_bookings, tmp_path)

    result = stage_timings(raw_dir, tmp_path / "output")

    assert set(result) == {"read_csv", "clean_bookings", "time_split", "write_splits", "input_schema"}
    assert all(value >= 0 for value in result.values())
    assert result["read_csv"] > 0


def test_profile_target_writes_json_and_profile(
    raw_bookings: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`profile_target` writes the JSON summary and the raw profile with the required fields."""
    raw_dir = write_raw(raw_bookings, tmp_path)
    monkeypatch.setattr("hotel_booking.profiling.PROFILE_DIR", tmp_path)

    json_path = profile_target(
        "preprocess",
        "before",
        Path("configs/model/logistic_regression.yaml"),
        runs=1,
        raw_dir=raw_dir,
        processed_dir=tmp_path / "processed",
    )

    assert (tmp_path / "preprocess-before.prof").exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert set(payload) == {
        "target",
        "label",
        "spec",
        "runs",
        "wall_seconds",
        "wall_median",
        "stages",
        "profiled_total",
        "functions",
        "environment",
    }
    assert payload["target"] == "preprocess"
    assert payload["spec"] is None
    assert payload["runs"] == 1
    assert payload["wall_median"] > 0
    assert payload["profiled_total"] > 0
    assert set(payload["stages"]) == {"read_csv", "clean_bookings", "time_split", "write_splits", "input_schema"}
    assert isinstance(payload["functions"], list)
    assert all(0.0 <= entry["share"] <= 1.0 for entry in payload["functions"])
    assert payload["environment"]["python"]


def test_profile_target_leaves_processed_dir_untouched(
    raw_bookings: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A preprocess profile never writes into the repository processed folder."""
    raw_dir = write_raw(raw_bookings, tmp_path)
    processed_dir = tmp_path / "repository" / "processed"
    monkeypatch.setattr("hotel_booking.profiling.PROFILE_DIR", tmp_path / "profile")

    profile_target(
        "preprocess",
        "before",
        Path("configs/model/logistic_regression.yaml"),
        runs=1,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
    )

    assert not processed_dir.exists()


def test_render_figures_writes_pngs(tmp_path: Path) -> None:
    """Figures are rendered when the before/after function lists differ in length and names."""
    profile_dir = tmp_path / "profile"
    stages = {"read_csv": 0.4, "clean_bookings": 0.3, "time_split": 0.1, "write_splits": 0.5, "input_schema": 0.01}
    write_pair(profile_dir, "preprocess", "preprocess", FUNCTIONS, OTHER_FUNCTIONS, stages)
    write_pair(profile_dir, "train-logistic_regression", "train", FUNCTIONS, FUNCTIONS)
    (profile_dir / "train-champion-before.json").write_text(
        json.dumps(summary("before", "train", 2.0, FUNCTIONS, {})), encoding="utf-8"
    )

    written = render_figures(profile_dir, tmp_path / "figures")

    assert [path.name for path in written] == ["profile_preprocess.png", "profile_train.png", "profile_summary.png"]
    assert all(path.stat().st_size > 0 for path in written)
    assert not (tmp_path / "figures" / "profile_champion.png").exists()


def test_render_figures_writes_champion_png(tmp_path: Path) -> None:
    """A complete champion pair produces the champion figure."""
    profile_dir = tmp_path / "profile"
    write_pair(profile_dir, "train-champion", "train", FUNCTIONS, OTHER_FUNCTIONS)

    written = render_figures(profile_dir, tmp_path / "figures")

    assert [path.name for path in written] == ["profile_champion.png", "profile_summary.png"]
    assert (tmp_path / "figures" / "profile_champion.png").stat().st_size > 0


def test_align_functions_ignores_line_moves() -> None:
    """A function that moved by a few lines still lines up before and after."""
    labels, before, after = _align_functions(
        [{"name": "fit (model.py:226)", "cumtime": 1.0, "share": 0.5}],
        [{"name": "fit (model.py:227)", "cumtime": 0.5, "share": 0.3}],
    )

    assert labels == ["fit (model.py)"]
    assert before == [1.0]
    assert after == [0.5]


def test_measure_training_runs_a_small_spec(raw_bookings: pd.DataFrame, tmp_path: Path) -> None:
    """`measure_training` fits a tiny spec once and reports a positive median."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    splits = data.time_split(data.clean_bookings(raw_bookings)[0])
    splits["train"].to_csv(processed_dir / "train.csv", index=False)
    splits["test"].to_csv(processed_dir / "test.csv", index=False)
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        "name: Logistic Regression\n"
        "encoding: onehot\n"
        "half_life_days: null\n"
        "booking_weights: false\n"
        "pipeline: {}\n"
        "params: {C: 1.0}\n",
        encoding="utf-8",
    )

    result = measure_training(spec_path, processed_dir, runs=1)

    assert result["runs"] == 1
    assert result["wall_median"] > 0
    assert result["stages"] == {}
