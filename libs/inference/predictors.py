"""Predictor protocol: a batch of normalised tiles in, probabilities out."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class Predictor(Protocol):
    def predict(self, batch: np.ndarray) -> np.ndarray:
        """(B, C, H, W) float32 -> (B, H, W) float32 probabilities in [0, 1]."""


class FakePredictor:
    """Constant output, for tests and plumbing checks."""

    def __init__(self, value: float = 0.5):
        self.value = value

    def predict(self, batch: np.ndarray) -> np.ndarray:
        b, _, h, w = batch.shape
        return np.full((b, h, w), self.value, dtype=np.float32)


class LocalTorchPredictor:
    """In-process PyTorch model (a LaunchSiteSegModule or any nn.Module returning logits)."""

    def __init__(self, model, device: str = "cpu"):
        import torch

        self._torch = torch
        self.model = model.to(device).eval()
        self.device = device

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str | Path, device: str = "cpu"
    ) -> LocalTorchPredictor:
        from services.training.lightning_module import LaunchSiteSegModule

        module = LaunchSiteSegModule.load_from_checkpoint(str(checkpoint_path), map_location=device)
        return cls(module, device=device)

    def predict(self, batch: np.ndarray) -> np.ndarray:
        torch = self._torch
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(batch, dtype=np.float32)).to(self.device)
            logits = self.model(x)
            return torch.sigmoid(logits).squeeze(1).cpu().numpy().astype(np.float32)


class TritonPredictor:
    """Remote model via a TritonSegClient-like object exposing infer_batch(images) -> probs."""

    def __init__(self, client):
        self.client = client

    @classmethod
    def connect(cls, url: str, model_name: str = "unet_seg") -> TritonPredictor:
        from services.serving.triton_client import TritonSegClient

        return cls(TritonSegClient(url=url, model_name=model_name))

    def predict(self, batch: np.ndarray) -> np.ndarray:
        return np.asarray(self.client.infer_batch(batch), dtype=np.float32)

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
