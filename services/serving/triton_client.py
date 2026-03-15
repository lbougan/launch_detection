"""Triton Inference Server gRPC client for UNet tile segmentation."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.grpc.aio as grpcclient_aio

from libs.features.indices import add_indices, normalize_percentile

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
            raise RuntimeError(
                f"Model {self.model_name}:{self.model_version} is not ready"
            )
        logger.info(
            "Connected to Triton at %s, model %s ready", self.url, self.model_name
        )

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


class AsyncTritonSegClient:
    """Async gRPC client for concurrent tile submission."""

    def __init__(
        self,
        url: str = DEFAULT_TRITON_URL,
        model_name: str = DEFAULT_MODEL_NAME,
        model_version: str = DEFAULT_MODEL_VERSION,
    ):
        self.url = url
        self.model_name = model_name
        self.model_version = model_version
        self._client = grpcclient_aio.InferenceServerClient(url=url)

    async def ensure_ready(self) -> None:
        if not await self._client.is_server_live():
            raise ConnectionError(f"Triton server at {self.url} is not live")
        if not await self._client.is_model_ready(self.model_name, self.model_version):
            raise RuntimeError(f"Model {self.model_name} not ready")

    async def infer_batch_async(self, images: np.ndarray) -> np.ndarray:
        """Async batch inference returning sigmoid probabilities."""
        images = np.ascontiguousarray(images, dtype=np.float32)

        inp = grpcclient_aio.InferInput("input", list(images.shape), "FP32")
        inp.set_data_from_numpy(images)

        out = grpcclient_aio.InferRequestedOutput("output")

        result = await self._client.infer(
            model_name=self.model_name,
            model_version=self.model_version,
            inputs=[inp],
            outputs=[out],
        )
        logits = result.as_numpy("output")
        return _sigmoid(logits.squeeze(1))

    async def close(self) -> None:
        await self._client.close()


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _tile_windows(
    height: int, width: int, tile_size: int, overlap: int
) -> list[tuple[int, int]]:
    """Generate (row, col) top-left coordinates for sliding windows."""
    step = tile_size - overlap
    windows = []
    for row in range(0, height - tile_size + 1, step):
        for col in range(0, width - tile_size + 1, step):
            windows.append((row, col))
    return windows


def _read_and_preprocess_tile(
    src,
    row: int,
    col: int,
    tile_size: int,
    use_indices: bool,
) -> np.ndarray:
    """Read a single tile from an open rasterio dataset and preprocess it."""
    from rasterio.windows import Window

    window = Window(col, row, tile_size, tile_size)
    data = src.read(window=window).astype(np.float32)
    data = normalize_percentile(data)
    if use_indices and data.shape[0] == 6:
        data = add_indices(data)
    return data


def _generate_tile_batches(
    src,
    windows: list[tuple[int, int]],
    tile_size: int,
    batch_size: int,
    use_indices: bool,
) -> Iterator[tuple[list[tuple[int, int]], np.ndarray]]:
    """Yield (coords, batch_array) tuples from an open rasterio dataset."""
    batch_images: list[np.ndarray] = []
    batch_coords: list[tuple[int, int]] = []

    for row, col in windows:
        tile = _read_and_preprocess_tile(src, row, col, tile_size, use_indices)
        batch_images.append(tile)
        batch_coords.append((row, col))

        if len(batch_images) == batch_size:
            yield batch_coords, np.stack(batch_images)
            batch_images = []
            batch_coords = []

    if batch_images:
        yield batch_coords, np.stack(batch_images)


def triton_sliding_window_inference(
    raster_path: Path,
    output_path: Path,
    triton_url: str = DEFAULT_TRITON_URL,
    model_name: str = DEFAULT_MODEL_NAME,
    tile_size: int = 256,
    overlap: int = 64,
    batch_size: int = 16,
    use_indices: bool = True,
) -> Path:
    """Sliding-window inference using Triton Inference Server.

    Drop-in replacement for ``services.training.inference.sliding_window_inference``
    with the model served remotely via Triton instead of loaded in-process.

    Returns:
        Path to the output probability raster (single-band float32 GeoTIFF).
    """
    import rasterio

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = TritonSegClient(url=triton_url, model_name=model_name)

    try:
        with rasterio.open(raster_path) as src:
            h, w = src.height, src.width
            profile = src.profile.copy()

            prob_map = np.zeros((h, w), dtype=np.float32)
            count_map = np.zeros((h, w), dtype=np.float32)

            windows = _tile_windows(h, w, tile_size, overlap)
            n_batches = 0

            for coords, batch_array in _generate_tile_batches(
                src, windows, tile_size, batch_size, use_indices
            ):
                probs = client.infer_batch(batch_array)

                for (row, col), tile_prob in zip(coords, probs):
                    prob_map[row : row + tile_size, col : col + tile_size] += tile_prob
                    count_map[row : row + tile_size, col : col + tile_size] += 1.0

                n_batches += 1
    finally:
        client.close()

    valid = count_map > 0
    prob_map[valid] /= count_map[valid]

    out_profile = profile.copy()
    out_profile.update(count=1, dtype="float32", compress="deflate")
    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(prob_map, 1)

    logger.info(
        "Triton inference complete: %s → %s (%d windows, %d batches)",
        raster_path,
        output_path,
        len(windows),
        n_batches,
    )
    return output_path
