from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.base import clone

from hotel_booking import config, model

MODEL_CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "model"
INPUTS = config.MODEL_INPUT_COLUMNS


@pytest.mark.parametrize(
    "name, options",
    [
        ("Logistic Regression", {}),
        ("Logistic Regression", {"log_skewed": True, "lead_time_bins": True, "crosses": True}),
        ("Random Forest", {"n_estimators": 20}),
        ("XGBoost", {"relative_adr": True, "n_estimators": 20}),
        ("LightGBM", {"n_estimators": 20}),
    ],
)
def test_models_train_and_handle_unseen_values(
    sample: pd.DataFrame, booking: dict[str, Any], name: str, options: dict[str, Any]
) -> None:
    """Every supported model fits raw bookings and predicts valid probabilities for unseen categories."""
    clf = model.make_model(name, half_life_days=180, **options)
    clf.fit(sample[INPUTS], sample[config.TARGET])
    unseen = pd.DataFrame([{**booking, "agent": 99999.0, "reserved_room_type": "Z", "meal": "Undefined"}])
    proba = clf.predict_proba(pd.concat([sample[INPUTS].head(50), unseen], ignore_index=True))[:, 1]
    assert np.all((proba >= 0) & (proba <= 1))


def test_model_is_cloneable_and_survives_pickling(sample: pd.DataFrame, tmp_path: Path) -> None:
    """A fitted model reloaded with joblib gives identical predictions."""
    clf = model.make_model("LightGBM", half_life_days=365, n_estimators=20)
    clone(clf)
    clf.fit(sample[INPUTS], sample[config.TARGET])
    path = tmp_path / "model.joblib"
    joblib.dump(clf, path)
    reloaded = joblib.load(path)
    np.testing.assert_allclose(reloaded.predict_proba(sample[INPUTS]), clf.predict_proba(sample[INPUTS]))


def test_recency_weighting_changes_the_model(sample: pd.DataFrame) -> None:
    """Recency weights change the fitted model."""
    plain = model.make_model("LightGBM", n_estimators=20).fit(sample[INPUTS], sample[config.TARGET])
    weighted = model.make_model("LightGBM", half_life_days=90, n_estimators=20).fit(
        sample[INPUTS], sample[config.TARGET]
    )
    assert not np.allclose(plain.predict_proba(sample[INPUTS]), weighted.predict_proba(sample[INPUTS]))


def test_log_transform_only_touches_skewed_columns(sample: pd.DataFrame) -> None:
    """log1p is applied to skewed counts only, not to binary flags."""
    clf = model.make_model("Logistic Regression", log_skewed=True).fit(sample[INPUTS], sample[config.TARGET])
    preprocess = clf.estimator_.named_steps["preprocess"]
    logged = preprocess.named_transformers_["log"].feature_names_in_
    assert set(logged) <= set(model.LOG_FEATURES)
    assert "is_family" not in logged and "lead_time" in logged


def test_booking_weights_count_each_booking_once(sample: pd.DataFrame) -> None:
    """With booking weights, a k-room booking weighs as much as a single-room booking."""
    frame = sample[INPUTS].head(4).assign(**{config.GROUP_SIZE_COLUMN: [1, 2, 2, 1]})
    weights = model.make_model("LightGBM", booking_weights=True).sample_weights(frame)
    assert weights is not None
    np.testing.assert_allclose(weights, [4 / 3, 2 / 3, 2 / 3, 4 / 3])


@pytest.mark.parametrize("config_name", ["champion", "logistic_regression"])
def test_model_configs_build_and_predict(sample: pd.DataFrame, config_name: str) -> None:
    """The model configs in configs/model build a model that fits raw bookings."""
    spec = yaml.safe_load((MODEL_CONFIGS / f"{config_name}.yaml").read_text(encoding="utf-8"))
    clf = model.build_from_spec(spec)
    proba = clf.fit(sample[INPUTS], sample[config.TARGET]).predict_proba(sample[INPUTS])[:, 1]
    assert np.all((proba >= 0) & (proba <= 1))
    if config_name == "champion":
        assert isinstance(clf, model.AveragingClassifier) and len(clf.estimators_) == 3


def test_build_from_spec_rejects_unknown_encoding_and_model() -> None:
    """Specs with an encoding or model this repo does not support fail early."""
    with pytest.raises(ValueError, match="encoding"):
        model.build_from_spec({"name": "LightGBM", "encoding": "native"})
    with pytest.raises(ValueError, match="model"):
        model.build_from_spec({"name": "CatBoost"})
