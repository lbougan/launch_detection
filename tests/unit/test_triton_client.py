"""TritonSegClient against a stubbed `tritonclient.grpc` (no server needed)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from services.serving import triton_client


class FakeResult:
    def __init__(self, logits: np.ndarray):
        self.logits = logits

    def as_numpy(self, name: str) -> np.ndarray:
        assert name == "output"
        return self.logits


class FakeServer:
    """Fails `failures` times, then returns zero logits shaped (B, 1, H, W)."""

    def __init__(self, url: str, failures: int = 0, live: bool = True, ready: bool = True):
        self.url, self.failures, self.live, self.ready = url, failures, live, ready
        self.calls = 0
        self.closed = False

    def is_server_live(self) -> bool:
        return self.live

    def is_model_ready(self, name: str, version: str) -> bool:
        return self.ready

    def infer(self, model_name, model_version, inputs, outputs):
        self.calls += 1
        if self.calls <= self.failures:
            raise ConnectionError("transient")
        b, _, h, w = inputs[0].data.shape
        return FakeResult(np.zeros((b, 1, h, w), dtype=np.float32))

    def close(self) -> None:
        self.closed = True


class FakeInput:
    def __init__(self, name, shape, dtype):
        self.name, self.shape, self.dtype = name, shape, dtype

    def set_data_from_numpy(self, data):
        self.data = data


def _stub(monkeypatch, **server_kw) -> dict:
    holder: dict = {}

    def make_server(url):
        holder["server"] = FakeServer(url, **server_kw)
        return holder["server"]

    fake = SimpleNamespace(
        InferenceServerClient=make_server,
        InferInput=FakeInput,
        InferRequestedOutput=lambda name: SimpleNamespace(name=name),
    )
    monkeypatch.setattr(triton_client, "grpcclient", fake)
    monkeypatch.setattr(
        triton_client.time, "sleep", lambda s: holder.setdefault("sleeps", []).append(s)
    )
    return holder


def test_infer_batch_returns_sigmoid_probabilities(monkeypatch):
    holder = _stub(monkeypatch)
    client = triton_client.TritonSegClient(url="triton:8001")
    probs = client.infer_batch(np.zeros((2, 10, 8, 8)))
    assert probs.shape == (2, 8, 8)
    np.testing.assert_allclose(probs, 0.5)
    assert holder["server"].url == "triton:8001" and "sleeps" not in holder
    client.close()
    assert holder["server"].closed


def test_infer_batch_retries_then_succeeds(monkeypatch):
    holder = _stub(monkeypatch, failures=2)
    client = triton_client.TritonSegClient(max_retries=3, backoff_base=0.5)
    assert client.infer_batch(np.zeros((1, 10, 4, 4))).shape == (1, 4, 4)
    assert holder["server"].calls == 3
    assert holder["sleeps"] == [0.5, 1.0], "exponential backoff"


def test_infer_batch_fails_after_retries(monkeypatch):
    holder = _stub(monkeypatch, failures=10)
    client = triton_client.TritonSegClient(max_retries=3)
    with pytest.raises(RuntimeError, match="after 3 retries") as exc:
        client.infer_batch(np.zeros((1, 10, 4, 4)))
    assert isinstance(exc.value.__cause__, ConnectionError)
    assert holder["server"].calls == 3


@pytest.mark.parametrize(
    ("kw", "err"), [({"live": False}, ConnectionError), ({"ready": False}, RuntimeError)]
)
def test_connect_checks_server_and_model(monkeypatch, kw, err):
    _stub(monkeypatch, **kw)
    with pytest.raises(err):
        triton_client.TritonSegClient()
