"""Inference: sliding-window prediction over AOIs, producing probability rasters."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window

from libs.features.indices import add_indices, normalize_percentile
from services.training.lightning_module import LaunchSiteSegModule

logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str | Path, device: str = "cpu") -> LaunchSiteSegModule:
    """Load a trained model from a Lightning checkpoint."""
    module = LaunchSiteSegModule.load_from_checkpoint(str(checkpoint_path), map_location=device)
    module.eval()
    module.to(device)
    return module


def predict_tile(
    model: LaunchSiteSegModule,
    image: np.ndarray,
    device: str = "cpu",
) -> np.ndarray:
    """Run inference on a single tile.

    Args:
        model: Loaded Lightning module.
        image: (C, H, W) float32 array (already normalized + indices).
        device: torch device.

    Returns:
        (H, W) float32 probability map.
    """
    tensor = torch.from_numpy(image[None]).float().to(device)
    with torch.no_grad():
        logits = model(tensor)
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    return probs


def sliding_window_inference(
    model: LaunchSiteSegModule,
    raster_path: Path,
    output_path: Path,
    tile_size: int = 256,
    overlap: int = 64,
    batch_size: int = 8,
    device: str = "cpu",
    use_indices: bool = True,
) -> Path:
    """Run sliding-window inference over a full raster.

    Produces a single-band probability raster (COG) of the same extent.

    Args:
        model: Loaded model.
        raster_path: Input multi-band raster.
        output_path: Where to write the probability raster.
        tile_size: Window size in pixels.
        overlap: Overlap between adjacent windows.
        batch_size: Number of windows to process at once.
        device: "cpu" or "cuda".
        use_indices: Whether to compute spectral indices before inference.

    Returns:
        Path to the output probability raster.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(raster_path) as src:
        h, w = src.height, src.width
        profile = src.profile.copy()

        prob_map = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)

        step = tile_size - overlap
        windows = []
        for row in range(0, h - tile_size + 1, step):
            for col in range(0, w - tile_size + 1, step):
                windows.append((row, col))

        batch_images = []
        batch_coords = []

        for row, col in windows:
            window = Window(col, row, tile_size, tile_size)
            data = src.read(window=window).astype(np.float32)
            data = normalize_percentile(data)

            if use_indices and data.shape[0] == 6:
                data = add_indices(data)

            batch_images.append(data)
            batch_coords.append((row, col))

            if len(batch_images) == batch_size:
                _process_batch(model, batch_images, batch_coords, prob_map, count_map, tile_size, device)
                batch_images = []
                batch_coords = []

        if batch_images:
            _process_batch(model, batch_images, batch_coords, prob_map, count_map, tile_size, device)

    valid = count_map > 0
    prob_map[valid] /= count_map[valid]

    out_profile = profile.copy()
    out_profile.update(count=1, dtype="float32", compress="deflate")
    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(prob_map, 1)

    logger.info("Inference complete: %s → %s (%d windows)", raster_path, output_path, len(windows))
    return output_path


def _process_batch(
    model: LaunchSiteSegModule,
    images: list[np.ndarray],
    coords: list[tuple[int, int]],
    prob_map: np.ndarray,
    count_map: np.ndarray,
    tile_size: int,
    device: str,
) -> None:
    """Run a batch of tiles through the model and accumulate results."""
    batch_tensor = torch.from_numpy(np.stack(images)).float().to(device)
    with torch.no_grad():
        logits = model(batch_tensor)
    probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

    for (row, col), tile_prob in zip(coords, probs):
        prob_map[row:row + tile_size, col:col + tile_size] += tile_prob
        count_map[row:row + tile_size, col:col + tile_size] += 1.0
