import hashlib
import json

import pandas as pd
import pytest
import yaml

from hotel_booking import config, data
from hotel_booking.evaluate import binary_metrics
from hotel_booking.model import build_from_spec
from tests.conftest import PROJECT_ROOT, RAW_PATH

CARD = json.loads((PROJECT_ROOT / "reports" / "it3190_model_card.json").read_text(encoding="utf-8"))
SPEC = yaml.safe_load((PROJECT_ROOT / "configs" / "model" / "champion.yaml").read_text(encoding="utf-8"))
TOLERANCE = 0.005


def test_champion_spec_matches_it3190_card() -> None:
    """The champion config is the spec recorded in the IT3190 model card."""
    assert SPEC == CARD["model_spec"]


def test_champion_reproduces_it3190_test_metrics(real_raw: pd.DataFrame) -> None:
    """Trained here on the same data, the champion scores within 0.005 of the IT3190 card on test (01-04/2017)."""
    assert hashlib.sha256(RAW_PATH.read_bytes()).hexdigest() == CARD["data"]["raw_sha256"]
    clean, _ = data.clean_bookings(real_raw)
    splits = data.time_split(clean)
    train, test = splits["train"], splits["test"]

    model = build_from_spec(SPEC).fit(train[config.MODEL_INPUT_COLUMNS], train[config.TARGET])
    proba = model.predict_proba(test[config.MODEL_INPUT_COLUMNS])[:, 1]
    metrics = binary_metrics(test[config.TARGET], proba, CARD["decision_threshold"])

    for name in ("roc_auc", "average_precision", "f1"):
        assert metrics[name] == pytest.approx(CARD["test_metrics"][name], abs=TOLERANCE), name
