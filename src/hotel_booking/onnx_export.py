"""Export the served model to ONNX and run it with ONNX Runtime (contract C8 in `docs/source/contracts.md`).

`BookingFeatureBuilder` is plain pandas code, so it stays in Python: only the part after it (imputation, log1p,
scaling, one-hot encoding and the classifier) becomes an ONNX graph. An ensemble exports one graph per member and
averages their probabilities, exactly like `model.AveragingClassifier`.
"""

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import pandas as pd
import typer
from lightgbm import LGBMClassifier
from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from skl2onnx.common.shape_calculator import (
    calculate_linear_classifier_output_shapes,
    calculate_linear_regressor_output_shapes,
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from skl2onnx.operator_converters.one_hot_encoder import convert_sklearn_one_hot_encoder
from skl2onnx.shape_calculators.one_hot_encoder import calculate_sklearn_one_hot_encoder_output_shapes
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from xgboost import XGBClassifier

from hotel_booking.config import MODEL_INPUT_COLUMNS
from hotel_booking.loader import CARD_FILE, MODEL_FILE, file_sha256
from hotel_booking.model import AveragingClassifier, RecencyWeightedClassifier, log1p_nonnegative

TARGET_OPSET = 17
ONNX_FILE = "model.onnx"
TEXT_FILL = "missing"


def _log1p_shape(operator: Any) -> None:
    """Output shape of `log1p_nonnegative`: same as its input."""
    calculate_linear_regressor_output_shapes(operator)
    operator.outputs[0].type = operator.inputs[0].type.__class__(operator.inputs[0].type.shape)


def _log1p_converter(scope: Any, operator: Any, container: Any) -> None:
    """ONNX nodes for `FunctionTransformer(log1p_nonnegative)`: log(1 + max(x, 0)).

    Raises:
        NotImplementedError: For any other function.
    """
    function = operator.raw_operator.func
    if function is not log1p_nonnegative:
        raise NotImplementedError(f"No ONNX converter for FunctionTransformer({function!r}).")
    zero = scope.get_unique_variable_name("zero")
    one = scope.get_unique_variable_name("one")
    container.add_initializer(zero, onnx.TensorProto.FLOAT, [], [0.0])
    container.add_initializer(one, onnx.TensorProto.FLOAT, [], [1.0])
    clipped = scope.get_unique_variable_name("clipped")
    shifted = scope.get_unique_variable_name("shifted")
    container.add_node("Max", [operator.inputs[0].full_name, zero], clipped, name=scope.get_unique_operator_name("Max"))
    container.add_node("Add", [clipped, one], shifted, name=scope.get_unique_operator_name("Add"))
    container.add_node("Log", [shifted], operator.outputs[0].full_name, name=scope.get_unique_operator_name("Log"))


def _has_infrequent(encoder: OneHotEncoder) -> bool:
    """Whether a fitted encoder grouped some categories as infrequent."""
    return any(categories is not None for categories in getattr(encoder, "infrequent_categories_", []) or [])


def _one_hot_shape(operator: Any) -> None:
    """Output shape of a one-hot encoder: one float column per output feature."""
    encoder = operator.raw_operator
    if not _has_infrequent(encoder):
        calculate_sklearn_one_hot_encoder_output_shapes(operator)
        return
    rows = operator.inputs[0].get_first_dimension()
    operator.outputs[0].type = FloatTensorType([rows, len(encoder.get_feature_names_out())])


def one_hot_codes(encoder: OneHotEncoder, feature: int) -> tuple[list[str], list[int], int, int]:
    """Output column of every known category of one feature, as sklearn's `transform` lays them out.

    Frequent categories keep their order; all infrequent ones share the last column. An unseen category goes to
    the infrequent column when it exists (`handle_unknown="infrequent_if_exist"`), else to no column.

    Args:
        encoder: A fitted encoder.
        feature: Input feature index.

    Returns:
        The categories, their output column, the column for unseen values (-1 for none) and the column count.
    """
    categories = [str(value) for value in encoder.categories_[feature]]
    rare = encoder.infrequent_categories_[feature]
    rare_set = {str(value) for value in rare} if rare is not None else set()
    frequent = [value for value in categories if value not in rare_set]
    width = len(frequent) + (1 if rare_set else 0)
    column = {value: index for index, value in enumerate(frequent)}
    codes = [column.get(value, len(frequent)) for value in categories]
    unseen = len(frequent) if rare_set and encoder.handle_unknown == "infrequent_if_exist" else -1
    return categories, codes, unseen, width


def _one_hot_converter(scope: Any, operator: Any, container: Any) -> None:
    """ONNX nodes for a text one-hot encoder, including scikit-learn's infrequent categories.

    Each input column goes through a LabelEncoder (category to output column) and is compared with the column
    indices; the per-feature blocks are concatenated. Encoders without infrequent categories use skl2onnx's own
    converter.

    Raises:
        NotImplementedError: For numeric inputs or `drop`, which the project's pipelines never use.
    """
    encoder = operator.raw_operator
    if not _has_infrequent(encoder):
        convert_sklearn_one_hot_encoder(scope, operator, container)
        return
    if encoder.drop is not None or not isinstance(operator.inputs[0].type, StringTensorType):
        raise NotImplementedError("Infrequent categories are converted only for text inputs without `drop`.")
    source = operator.inputs[0].full_name
    blocks = []
    for feature in range(len(encoder.categories_)):
        categories, codes, unseen, width = one_hot_codes(encoder, feature)
        index = scope.get_unique_variable_name(f"feature{feature}")
        container.add_initializer(index, onnx.TensorProto.INT64, [1], [feature])
        column = scope.get_unique_variable_name(f"column{feature}")
        container.add_node("Gather", [source, index], column, axis=1, name=scope.get_unique_operator_name("Gather"))
        code = scope.get_unique_variable_name(f"code{feature}")
        container.add_node(
            "LabelEncoder",
            [column],
            code,
            op_domain="ai.onnx.ml",
            op_version=2,
            keys_strings=categories,
            values_int64s=codes,
            default_int64=unseen,
            name=scope.get_unique_operator_name("LabelEncoder"),
        )
        positions = scope.get_unique_variable_name(f"positions{feature}")
        container.add_initializer(positions, onnx.TensorProto.INT64, [width], list(range(width)))
        hits = scope.get_unique_variable_name(f"hits{feature}")
        container.add_node("Equal", [code, positions], hits, name=scope.get_unique_operator_name("Equal"))
        block = scope.get_unique_variable_name(f"onehot{feature}")
        container.add_node(
            "Cast", [hits], block, to=onnx.TensorProto.FLOAT, name=scope.get_unique_operator_name("Cast")
        )
        blocks.append(block)
    container.add_node(
        "Concat", blocks, operator.outputs[0].full_name, axis=1, name=scope.get_unique_operator_name("Concat")
    )


def register_converters() -> None:
    """Teach skl2onnx the custom log1p step, infrequent one-hot categories, LightGBM and XGBoost."""
    update_registered_converter(
        FunctionTransformer, "HotelBookingLog1p", _log1p_shape, _log1p_converter, overwrite=True
    )
    update_registered_converter(
        OneHotEncoder, "SklearnOneHotEncoder", _one_hot_shape, _one_hot_converter, overwrite=True
    )
    update_registered_converter(
        LGBMClassifier,
        "LightGbmLGBMClassifier",
        calculate_linear_classifier_output_shapes,
        convert_lightgbm,
        options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
        overwrite=True,
    )
    update_registered_converter(
        XGBClassifier,
        "XGBoostXGBClassifier",
        calculate_linear_classifier_output_shapes,
        convert_xgboost,
        options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
        overwrite=True,
    )


def members_of(model: Any) -> list[RecencyWeightedClassifier]:
    """The fitted single models inside a served model.

    Args:
        model: A fitted `RecencyWeightedClassifier` or `AveragingClassifier`.

    Returns:
        The single models, one per ONNX graph.
    """
    if isinstance(model, AveragingClassifier):
        return list(model.estimators_)
    if isinstance(model, RecencyWeightedClassifier):
        return [model]
    raise TypeError(f"Cannot export {type(model).__name__} to ONNX.")


def is_text(column: pd.Series) -> bool:
    """Whether a feature column holds text (categorical) values."""
    return bool(pd.api.types.is_string_dtype(column) or column.dtype == object)


def feature_inputs(features: pd.DataFrame) -> dict[str, np.ndarray]:
    """ONNX Runtime inputs: one (n, 1) array per feature column, text as str and numbers as float32.

    Args:
        features: Output of `BookingFeatureBuilder.transform`.

    Returns:
        Input name to array.
    """
    inputs = {}
    for name in features.columns:
        column = features[name]
        if is_text(column):
            inputs[name] = column.astype(object).where(column.notna(), TEXT_FILL).astype(str).to_numpy().reshape(-1, 1)
        else:
            inputs[name] = column.to_numpy(dtype=np.float32, na_value=np.nan).reshape(-1, 1)
    return inputs


def prepare_for_conversion(pipeline: Pipeline) -> Pipeline:
    """Adjust a copy of the fitted steps so the converters accept them, without changing predictions.

    - Constant imputers of text columns: `feature_inputs` already replaces missing text with `TEXT_FILL`, their
      fill value, so in the graph they only need to map `TEXT_FILL` to itself.
    - XGBoost: the converter needs the booster's default `f0, f1...` feature names; the inputs are positional, so
      dropping the pandas column names changes nothing.

    Args:
        pipeline: A deep copy of the steps after `BookingFeatureBuilder`.

    Returns:
        The same pipeline, adjusted in place.
    """
    for estimator in _walk(pipeline):
        if (
            isinstance(estimator, SimpleImputer)
            and estimator.strategy == "constant"
            and estimator.fill_value == TEXT_FILL
        ):
            estimator.missing_values = TEXT_FILL
        if isinstance(estimator, XGBClassifier):
            estimator.get_booster().feature_names = None
    return pipeline


def _walk(estimator: Any) -> list[Any]:
    """The estimator and every estimator nested in its pipelines and column transformers."""
    found = [estimator]
    if isinstance(estimator, Pipeline):
        for _, step in estimator.steps:
            found += _walk(step)
    elif isinstance(estimator, ColumnTransformer):
        for _, transformer, _ in estimator.transformers_:
            found += _walk(transformer)
    return found


def convert_member(member: RecencyWeightedClassifier, sample: pd.DataFrame) -> onnx.ModelProto:
    """Convert the steps after `BookingFeatureBuilder` of one fitted single model.

    Args:
        member: The fitted single model.
        sample: A few raw bookings, used to read the feature types.

    Returns:
        The ONNX graph; it takes one input per feature and its second output holds the probabilities, (n, 2).
    """
    register_converters()
    pipeline = member.estimator_
    features = pipeline.named_steps["features"].transform(sample[MODEL_INPUT_COLUMNS])
    initial_types = [
        (name, StringTensorType([None, 1]) if is_text(features[name]) else FloatTensorType([None, 1]))
        for name in features.columns
    ]
    tail = prepare_for_conversion(Pipeline(copy.deepcopy(pipeline.steps[1:])))
    graph = convert_sklearn(
        tail,
        initial_types=initial_types,
        target_opset={"": TARGET_OPSET, "ai.onnx.ml": 3},
        options={id(tail.steps[-1][1]): {"zipmap": False}},
    )
    return graph


class OnnxModel:
    """Predicts like the joblib model: Python feature builders followed by ONNX Runtime graphs."""

    def __init__(self, model: Any, graphs: list[bytes]) -> None:
        """Pair each single model's feature builder with its ONNX graph.

        Args:
            model: The fitted joblib model, used only for its feature builders.
            graphs: Serialized ONNX graphs, one per single model, in `members_of` order.
        """
        import onnxruntime

        members = members_of(model)
        if len(members) != len(graphs):
            raise ValueError(f"The model has {len(members)} members but {len(graphs)} ONNX graphs were given.")
        options = onnxruntime.SessionOptions()
        options.log_severity_level = 3
        self.builders = [member.estimator_.named_steps["features"] for member in members]
        self.sessions = [
            onnxruntime.InferenceSession(graph, options, providers=["CPUExecutionProvider"]) for graph in graphs
        ]
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Class probabilities, averaged over the members.

        Args:
            X: Raw bookings with the columns in `MODEL_INPUT_COLUMNS`.

        Returns:
            Probabilities of shape (n_rows, 2).
        """
        outputs = [
            session.run([session.get_outputs()[1].name], feature_inputs(builder.transform(X)))[0]
            for builder, session in zip(self.builders, self.sessions, strict=True)
        ]
        return np.mean(outputs, axis=0).astype(np.float64)


def onnx_file_names(count: int) -> list[str]:
    """File names of the ONNX graphs: `model.onnx`, or `model.0.onnx`, `model.1.onnx`... for an ensemble."""
    return [ONNX_FILE] if count == 1 else [f"model.{index}.onnx" for index in range(count)]


def export_onnx(model: Any, sample: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Write the ONNX graphs of a fitted model.

    Args:
        model: The fitted joblib model.
        sample: A few raw bookings, used to read the feature types.
        output_dir: Folder for the graphs.

    Returns:
        The written files.
    """
    members = members_of(model)
    paths = [output_dir / name for name in onnx_file_names(len(members))]
    for member, path in zip(members, paths, strict=True):
        tmp = path.with_name(f"{path.name}.tmp")
        tmp.write_bytes(convert_member(member, sample).SerializeToString())
        tmp.replace(path)
    return paths


def max_abs_difference(model: Any, onnx_model: OnnxModel, frame: pd.DataFrame) -> float:
    """Largest absolute difference between the joblib and ONNX cancellation probabilities.

    Args:
        model: The joblib model.
        onnx_model: Its ONNX version.
        frame: Raw bookings.

    Returns:
        The maximum absolute difference.
    """
    expected = model.predict_proba(frame[MODEL_INPUT_COLUMNS])[:, 1]
    actual = onnx_model.predict_proba(frame[MODEL_INPUT_COLUMNS])[:, 1]
    return float(np.max(np.abs(expected - actual)))


def main(
    model_dir: Path = Path("models"), processed_dir: Path = Path("data/processed"), check_rows: int = 5000
) -> None:
    """Export `models/model.joblib` to ONNX, check it against the joblib model and record it in the card.

    Args:
        model_dir: Folder with `model.joblib` and `model_card.json`.
        processed_dir: Folder with `test.csv`, used for the parity check.
        check_rows: Test rows used for the parity check.
    """
    import joblib

    from hotel_booking.data import read_split

    model = joblib.load(model_dir / MODEL_FILE)
    check = read_split(processed_dir, "test").head(check_rows)
    paths = export_onnx(model, check.head(50), model_dir)
    onnx_model = OnnxModel(model, [path.read_bytes() for path in paths])
    difference = max_abs_difference(model, onnx_model, check)
    card_path = model_dir / CARD_FILE
    card = json.loads(card_path.read_text(encoding="utf-8"))
    card["onnx"] = {
        "artifacts": [{"file": path.name, "sha256": file_sha256(path)} for path in paths],
        "opset": TARGET_OPSET,
        "parity_rows": len(check),
        "parity_max_abs_difference": difference,
    }
    tmp = card_path.with_name(f"{card_path.name}.tmp")
    tmp.write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")
    tmp.replace(card_path)
    print(f"Wrote {', '.join(path.name for path in paths)}; max |p_joblib - p_onnx| = {difference:.2e}")


if __name__ == "__main__":
    typer.run(main)
