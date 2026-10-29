import numpy as np
import pytest
import torch

from libs.inference.predictors import FakePredictor, LocalTorchPredictor, TritonPredictor


def test_fake_predictor_shape():
    p = FakePredictor(value=0.3)
    out = p.predict(np.zeros((2, 10, 16, 16), dtype=np.float32))
    assert out.shape == (2, 16, 16) and float(out.mean()) == pytest.approx(0.3)


def test_local_predictor_wraps_a_module():
    class Tiny(torch.nn.Module):
        def forward(self, x):
            assert not torch.is_grad_enabled()
            assert not self.training
            return x[:, :1] * 0.0 + 2.0  # logits 2 -> sigmoid 0.88

    p = LocalTorchPredictor(Tiny(), device="cpu")
    out = p.predict(np.zeros((1, 10, 8, 8), dtype=np.float32))
    assert out.shape == (1, 8, 8)
    assert float(out[0, 0, 0]) == pytest.approx(1 / (1 + np.exp(-2.0)), rel=1e-4)


class _StubClient:
    def __init__(self):
        self.calls = []

    def infer_batch(self, images):
        self.calls.append(images.shape)
        return np.full(images.shape[:1] + images.shape[2:], 0.25, dtype=np.float32)


def test_triton_predictor_delegates_to_client():
    client = _StubClient()
    p = TritonPredictor(client)
    out = p.predict(np.zeros((3, 10, 8, 8), dtype=np.float32))
    assert out.shape == (3, 8, 8) and client.calls == [(3, 10, 8, 8)]
