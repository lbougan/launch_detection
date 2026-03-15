"""Memory-optimized large-area inference using Triton + memmap accumulation."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from libs.features.indices import add_indices, normalize_percentile
from services.serving.triton_client import (
    DEFAULT_MODEL_NAME,
    DEFAULT_TRITON_URL,
    TritonSegClient,
)

logger = logging.getLogger(__name__)

BYTES_PER_PIXEL_F32 = 4
TILE_CHANNELS = 10


def _estimate_strip_height(
    raster_width: int,
    tile_size: int,
    overlap: int,
    batch_size: int,
    max_memory_mb: float,
) -> int:
    """Choose a strip height that keeps peak memory under budget.

    Memory components per strip:
    - prob_map strip: strip_height * raster_width * 4 bytes
    - count_map strip: strip_height * raster_width * 4 bytes
    - one batch of tiles: batch_size * TILE_CHANNELS * tile_size * tile_size * 4 bytes
    """
    batch_bytes = batch_size * TILE_CHANNELS * tile_size * tile_size * BYTES_PER_PIXEL_F32
    budget_bytes = max_memory_mb * 1024 * 1024
    available_for_maps = budget_bytes - batch_bytes

    if available_for_maps <= 0:
        return tile_size

    bytes_per_row = 2 * raster_width * BYTES_PER_PIXEL_F32  # prob + count
    max_rows = int(available_for_maps / bytes_per_row)

    step = tile_size - overlap
    strip_h = max(tile_size, (max_rows // step) * step + overlap)
    return strip_h


def _create_memmap(shape: tuple[int, ...], tmpdir: str, name: str) -> np.ndarray:
    """Create a zero-initialized memory-mapped float32 array."""
    path = Path(tmpdir) / f"{name}.dat"
    mm = np.memmap(str(path), dtype=np.float32, mode="w+", shape=shape)
    mm[:] = 0.0
    return mm


def memory_optimized_inference(
    raster_path: Path,
    output_path: Path,
    triton_url: str = DEFAULT_TRITON_URL,
    model_name: str = DEFAULT_MODEL_NAME,
    tile_size: int = 256,
    overlap: int = 64,
    batch_size: int = 16,
    use_indices: bool = True,
    max_memory_mb: float = 512.0,
) -> Path:
    """Sliding-window Triton inference with bounded memory for large rasters.

    Uses memory-mapped arrays for the probability/count accumulation buffers
    and processes the raster in horizontal strips to cap peak RAM usage.

    Args:
        raster_path: Input multi-band raster (GeoTIFF).
        output_path: Where to write the single-band probability raster.
        triton_url: Triton gRPC endpoint.
        model_name: Model name in the Triton repository.
        tile_size: Tile height/width in pixels.
        overlap: Overlap between adjacent tiles.
        batch_size: Tiles per Triton request.
        use_indices: Compute spectral indices before inference.
        max_memory_mb: Approximate memory budget (MB) for maps + batch.

    Returns:
        Path to the output probability raster.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = TritonSegClient(url=triton_url, model_name=model_name)
    step = tile_size - overlap

    try:
        with rasterio.open(raster_path) as src:
            h, w = src.height, src.width
            profile = src.profile.copy()

            strip_height = _estimate_strip_height(
                w, tile_size, overlap, batch_size, max_memory_mb
            )
            logger.info(
                "Raster %dx%d, strip_height=%d, memory_budget=%.0f MB",
                w,
                h,
                strip_height,
                max_memory_mb,
            )

            with tempfile.TemporaryDirectory(prefix="orbiteye_") as tmpdir:
                prob_map = _create_memmap((h, w), tmpdir, "prob")
                count_map = _create_memmap((h, w), tmpdir, "count")

                strip_starts = list(range(0, h, strip_height - overlap))

                for strip_idx, strip_row0 in enumerate(strip_starts):
                    strip_row1 = min(strip_row0 + strip_height, h)
                    actual_strip_h = strip_row1 - strip_row0

                    if actual_strip_h < tile_size:
                        break

                    logger.info(
                        "Strip %d/%d: rows %d–%d",
                        strip_idx + 1,
                        len(strip_starts),
                        strip_row0,
                        strip_row1,
                    )

                    batch_images: list[np.ndarray] = []
                    batch_coords: list[tuple[int, int]] = []

                    for row in range(strip_row0, strip_row1 - tile_size + 1, step):
                        for col in range(0, w - tile_size + 1, step):
                            window = Window(col, row, tile_size, tile_size)
                            data = src.read(window=window).astype(np.float32)
                            data = normalize_percentile(data)
                            if use_indices and data.shape[0] == 6:
                                data = add_indices(data)

                            batch_images.append(data)
                            batch_coords.append((row, col))

                            if len(batch_images) == batch_size:
                                _accumulate_batch(
                                    client,
                                    batch_images,
                                    batch_coords,
                                    prob_map,
                                    count_map,
                                    tile_size,
                                )
                                batch_images = []
                                batch_coords = []

                    if batch_images:
                        _accumulate_batch(
                            client,
                            batch_images,
                            batch_coords,
                            prob_map,
                            count_map,
                            tile_size,
                        )

                valid = count_map > 0
                prob_map[valid] /= count_map[valid]

                out_profile = profile.copy()
                out_profile.update(count=1, dtype="float32", compress="deflate")
                with rasterio.open(output_path, "w", **out_profile) as dst:
                    dst.write(prob_map, 1)

    finally:
        client.close()

    logger.info("Memory-optimized inference complete: %s → %s", raster_path, output_path)
    return output_path


def _accumulate_batch(
    client: TritonSegClient,
    images: list[np.ndarray],
    coords: list[tuple[int, int]],
    prob_map: np.ndarray,
    count_map: np.ndarray,
    tile_size: int,
) -> None:
    """Run a batch through Triton and accumulate into the memmap arrays."""
    batch_array = np.stack(images)
    probs = client.infer_batch(batch_array)

    for (row, col), tile_prob in zip(coords, probs):
        prob_map[row : row + tile_size, col : col + tile_size] += tile_prob
        count_map[row : row + tile_size, col : col + tile_size] += 1.0
