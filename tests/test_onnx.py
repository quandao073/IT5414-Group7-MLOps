import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import pytest
import yaml
from fastapi.testclient import TestClient
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from hotel_booking import benchmark, config, data, onnx_export, sample_model
from hotel_booking.api import create_app
from hotel_booking.loader import FileSource, ModelLoadError, served_from_folder
from hotel_booking.model import build_from_spec

pytest.importorskip("onnxruntime", reason="onnxruntime has no wheel for Intel macOS (see pyproject.toml)")

MODEL_CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "model"
TOLERANCE = 1e-5


def fit_spec(name: str, sample: pd.DataFrame) -> Any:
    """Fit a model config on the synthetic train split."""
    spec = yaml.safe_load((MODEL_CONFIGS / f"{name}.yaml").read_text(encoding="utf-8"))
    train = data.time_split(sample)["train"]
    return build_from_spec(spec).fit(train[config.MODEL_INPUT_COLUMNS], train[config.TARGET])


@pytest.fixture(scope="module")
def exported(sample: pd.DataFrame, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A logistic regression saved like `invoke sample-model` and exported like `invoke export-onnx`."""
    folder = tmp_path_factory.mktemp("onnx")
    splits = data.time_split(sample)
    spec = yaml.safe_load((MODEL_CONFIGS / "logistic_regression.yaml").read_text(encoding="utf-8"))
    model, card = sample_model.train_with_card(spec, splits["train"], splits["test"], "LogReg", "onnx-test")
    sample_model.save_artifacts(model, card, folder)
    processed = tmp_path_factory.mktemp("processed")
    splits["test"].to_csv(processed / "test.csv", index=False)
    onnx_export.main(model_dir=folder, processed_dir=processed, check_rows=500)
    return folder


@pytest.mark.parametrize("name", ["logistic_regression", "champion"])
def test_onnx_matches_joblib(sample: pd.DataFrame, tmp_path: Path, name: str) -> None:
    """Both model configs export to ONNX and predict the same probabilities as the joblib model."""
    model = fit_spec(name, sample)
    paths = onnx_export.export_onnx(model, sample.head(20), tmp_path)
    assert len(paths) == len(onnx_export.members_of(model))
    onnx_model = onnx_export.OnnxModel(model, [path.read_bytes() for path in paths])
    holdout = data.time_split(sample)["production_holdout"]
    assert onnx_export.max_abs_difference(model, onnx_model, holdout) < TOLERANCE


def test_export_records_graphs_in_the_card(exported: Path) -> None:
    """The export writes model.onnx and lists it with its checksum and parity in the card."""
    card = json.loads((exported / sample_model.CARD_FILE).read_text(encoding="utf-8"))
    assert [artifact["file"] for artifact in card["onnx"]["artifacts"]] == [onnx_export.ONNX_FILE]
    assert card["onnx"]["parity_max_abs_difference"] < TOLERANCE
    assert not list(exported.glob("*.tmp"))


def test_onnx_handles_missing_and_unseen_values(exported: Path, booking: dict[str, Any]) -> None:
    """Missing numbers and unseen categories give the same probability in both backends."""
    joblib_model = served_from_folder(exported / "model.joblib", exported / "model_card.json", "joblib")
    onnx_model = served_from_folder(exported / "model.joblib", exported / "model_card.json", "onnx")
    odd = pd.DataFrame(
        [{**booking, "children": None, "agent": None, "hotel": "Airport Hotel", "meal": "Undefined"}, booking]
    )[config.MODEL_INPUT_COLUMNS]
    assert onnx_model.predict(odd) == pytest.approx(joblib_model.predict(odd), abs=TOLERANCE)


def test_api_serves_the_onnx_backend(exported: Path, booking: dict[str, Any]) -> None:
    """With the ONNX backend the API answers like the joblib backend and reports the backend in metrics."""
    responses = {}
    for backend in ("joblib", "onnx"):
        source = FileSource(exported / "model.joblib", exported / "model_card.json", backend)
        with TestClient(create_app(source=source)) as client:
            responses[backend] = client.post("/predict", json=booking).json()
            metrics = client.get("/metrics").text
        assert f'backend="{backend}"' in metrics
    assert responses["onnx"]["probability"] == pytest.approx(responses["joblib"]["probability"], abs=TOLERANCE)


def test_onnx_backend_needs_matching_graphs(exported: Path, tmp_path: Path) -> None:
    """Without an export, or with a changed graph, the ONNX backend refuses to load."""
    model = joblib.load(exported / "model.joblib")
    card = json.loads((exported / "model_card.json").read_text(encoding="utf-8"))
    sample_model.save_artifacts(model, {key: value for key, value in card.items() if key != "onnx"}, tmp_path / "a")
    with pytest.raises(ModelLoadError, match="no ONNX export"):
        served_from_folder(tmp_path / "a" / "model.joblib", tmp_path / "a" / "model_card.json", "onnx")
    changed = shutil.copytree(exported, tmp_path / "b")
    (changed / onnx_export.ONNX_FILE).write_bytes(b"changed")
    with pytest.raises(ModelLoadError, match="sha256"):
        served_from_folder(changed / "model.joblib", changed / "model_card.json", "onnx")


def test_one_hot_codes_follow_the_sklearn_layout() -> None:
    """Frequent categories keep their columns, rare and unseen ones share the infrequent column."""
    encoder = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=2, sparse_output=False)
    encoder.fit([["a"], ["a"], ["b"], ["b"], ["c"]])
    categories, codes, unseen, width = onnx_export.one_hot_codes(encoder, 0)
    assert dict(zip(categories, codes, strict=True)) == {"a": 0, "b": 1, "c": 2}
    assert unseen == 2 and width == 3
    assert list(encoder.get_feature_names_out()) == ["x0_a", "x0_b", "x0_infrequent_sklearn"]


def test_unknown_function_transformer_is_rejected() -> None:
    """Only the project's log1p step has an ONNX converter."""
    operator = type("Operator", (), {"raw_operator": FunctionTransformer(abs)})()
    with pytest.raises(NotImplementedError, match="FunctionTransformer"):
        onnx_export._log1p_converter(None, operator, None)


def test_benchmark_reports_both_backends(exported: Path, sample: pd.DataFrame) -> None:
    """The benchmark times both backends and reports sizes and parity."""
    result = benchmark.run_benchmark(exported, sample, single_repeats=5, batch_rows=50, batch_repeats=3)
    assert set(result["backends"]) == {"joblib", "onnx"}
    for values in result["backends"].values():
        assert values["single"]["p50_ms"] > 0 and values["artifact_mb"] > 0
    assert result["parity_max_abs_difference"] < TOLERANCE
    assert "| onnx |" in benchmark.to_markdown(result)
