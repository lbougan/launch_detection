"""Triton Inference Server gRPC client for UNet tile segmentation."""

from __future__ import annotations

import logging
import time

import numpy as np
import tritonclient.grpc as grpcclient

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "unet_seg"
DEFAULT_MODEL_VERSION = ""
DEFAULT_TRITON_URL = "localhost:8001"


class TritonSegClient:
    """Synchronous gRPC client for the UNet segmentation model on Triton.

    Handles connection management, input/output tensor construction,
    and optional retry with exponential backoff.
    """

    def __init__(
        self,
        url: str = DEFAULT_TRITON_URL,
        model_name: str = DEFAULT_MODEL_NAME,
        model_version: str = DEFAULT_MODEL_VERSION,
        max_retries: int = 3,
        backoff_base: float = 0.5,
    ):
        self.url = url
        self.model_name = model_name
        self.model_version = model_version
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._client = grpcclient.InferenceServerClient(url=url)
        self._ensure_model_ready()

    def _ensure_model_ready(self) -> None:
        if not self._client.is_server_live():
            raise ConnectionError(f"Triton server at {self.url} is not live")
        if not self._client.is_model_ready(self.model_name, self.model_version):
            raise RuntimeError(f"Model {self.model_name}:{self.model_version} is not ready")
        logger.info("Connected to Triton at %s, model %s ready", self.url, self.model_name)

    def infer_batch(self, images: np.ndarray) -> np.ndarray:
        """Send a batch of tiles and return sigmoid probabilities.

        Args:
            images: ``(B, 10, 256, 256)`` float32 array.

        Returns:
            ``(B, 256, 256)`` float32 probability map (sigmoid already applied).
        """
        images = np.ascontiguousarray(images, dtype=np.float32)

        inp = grpcclient.InferInput("input", list(images.shape), "FP32")
        inp.set_data_from_numpy(images)

        out = grpcclient.InferRequestedOutput("output")

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                result = self._client.infer(
                    model_name=self.model_name,
                    model_version=self.model_version,
                    inputs=[inp],
                    outputs=[out],
                )
                logits = result.as_numpy("output")  # (B, 1, H, W)
                probs = _sigmoid(logits.squeeze(1))  # (B, H, W)
                return probs
            except Exception as exc:
                last_exc = exc
                wait = self.backoff_base * (2**attempt)
                logger.warning(
                    "Triton infer attempt %d/%d failed: %s (retry in %.1fs)",
                    attempt + 1,
                    self.max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"Triton inference failed after {self.max_retries} retries"
        ) from last_exc

    def close(self) -> None:
        self._client.close()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
