import httpx
import pytest

import tasks

METRICS = """# HELP api_model_info Backend and model version served by the API.
# TYPE api_model_info gauge
api_model_info{backend="onnx",model_version="logistic_regression-20261113144119"} 1.0
"""


class FakeResponse:
    def __init__(self, payload: dict[str, object] | None = None, text: str = "", status_code: int = 200) -> None:
        self._payload = payload
        self.text = text
        self.status_code = status_code

    def json(self) -> dict[str, object]:
        assert self._payload is not None
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_parse_backend_reads_the_metric_label() -> None:
    assert tasks.parse_backend(METRICS) == "onnx"


def test_parse_backend_returns_none_without_the_metric() -> None:
    assert tasks.parse_backend("# no metrics here") is None


def test_wait_for_api_retries_until_the_model_is_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        FakeResponse({"status": "ok", "model_loaded": False}),
        FakeResponse({"status": "ok", "model_loaded": True}),
    ]
    calls: list[str] = []

    def fake_get(url: str, timeout: float) -> FakeResponse:
        calls.append(url)
        return responses.pop(0)

    monkeypatch.setattr(tasks.httpx, "get", fake_get)
    assert tasks.wait_for_api("http://api", timeout=1.0, interval=0.0) == {"status": "ok", "model_loaded": True}
    assert calls == ["http://api/health", "http://api/health"]


def test_wait_for_api_raises_when_the_api_stays_down(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_get(url: str, timeout: float) -> FakeResponse:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(tasks.httpx, "get", fake_get)
    with pytest.raises(RuntimeError, match="not ready"):
        tasks.wait_for_api("http://api", timeout=0.05, interval=0.0)


def test_verify_backend_accepts_the_requested_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tasks.httpx, "get", lambda url, timeout: FakeResponse(text=METRICS))
    assert tasks.verify_backend("http://api", "onnx") == "onnx"


def test_verify_backend_rejects_a_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tasks.httpx, "get", lambda url, timeout: FakeResponse(text='api_model_info{backend="joblib"} 1.0')
    )
    with pytest.raises(RuntimeError, match="joblib"):
        tasks.verify_backend("http://api", "onnx")
