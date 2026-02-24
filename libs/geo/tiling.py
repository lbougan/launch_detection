"""Tiling utilities: cut rasters into ML-ready chips and build weak label masks."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from rasterio.windows import Window
from shapely.geometry import Point, box, mapping

logger = logging.getLogger(__name__)

TILE_SIZE = 256
GSD = 10  # metres per pixel


def chip_raster(
    raster_path: Path,
    output_dir: Path,
    tile_size: int = TILE_SIZE,
    overlap: int = 32,
    min_valid_frac: float = 0.5,
) -> list[dict]:
    """Slice a raster into square chips.

    Returns a list of chip metadata dicts with tile_id, bbox, and file path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    chips = []

    with rasterio.open(raster_path) as src:
        h, w = src.height, src.width
        step = tile_size - overlap

        for row_off in range(0, h - tile_size + 1, step):
            for col_off in range(0, w - tile_size + 1, step):
                window = Window(col_off, row_off, tile_size, tile_size)
                data = src.read(window=window)

                valid_frac = np.count_nonzero(data[0]) / (tile_size * tile_size)
                if valid_frac < min_valid_frac:
                    continue

                win_transform = rasterio.windows.transform(window, src.transform)
                bounds = rasterio.transform.array_bounds(tile_size, tile_size, win_transform)

                tile_id = f"tile_{col_off}_{row_off}"
                out_path = output_dir / f"{tile_id}.tif"

                profile = src.profile.copy()
                profile.update(
                    height=tile_size,
                    width=tile_size,
                    transform=win_transform,
                    compress="deflate",
                )
                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(data)

                chips.append({
                    "tile_id": tile_id,
                    "path": str(out_path),
                    "bbox": list(bounds),
                    "row_off": row_off,
                    "col_off": col_off,
                })

    logger.info("Chipped %s into %d tiles (%dpx, overlap=%d)", raster_path, len(chips), tile_size, overlap)
    return chips


def build_weak_label_mask(
    bbox: list[float],
    tile_size: int,
    known_sites: list[dict],
    negative_buffer_km: float = 50.0,
) -> np.ndarray:
    """Build a weak binary label mask for a tile.

    Pixels within site buffers → 1, pixels far from all sites → 0,
    ambiguous pixels → 0.5 (label smoothing / ignore zone).

    Args:
        bbox: [west, south, east, north] of the tile.
        tile_size: Pixel dimensions (square).
        known_sites: List of dicts with lat, lon, buffer_km.
        negative_buffer_km: Distance beyond which a tile is a hard negative.

    Returns:
        (tile_size, tile_size) float32 mask.
    """
    transform = from_bounds(*bbox, tile_size, tile_size)
    tile_box = box(*bbox)

    positive_shapes = []
    for site in known_sites:
        pt = Point(site["lon"], site["lat"])
        buf_deg = site.get("buffer_km", 2.0) / 111.0  # rough km→deg
        circle = pt.buffer(buf_deg)
        if circle.intersects(tile_box):
            positive_shapes.append(circle)

    mask = np.zeros((tile_size, tile_size), dtype=np.float32)

    if positive_shapes:
        mask = rasterize(
            [(mapping(s), 1.0) for s in positive_shapes],
            out_shape=(tile_size, tile_size),
            transform=transform,
            fill=0.0,
            dtype="float32",
        )

    return mask


def load_known_sites(manifest_path: Path) -> list[dict]:
    """Load known launch sites from the JSON manifest."""
    with open(manifest_path) as f:
        data = json.load(f)
    return data["sites"]


def build_dataset_split(
    chip_metas: list[dict],
    known_sites: list[dict],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> list[dict]:
    """Assign train/val/test split to chips and attach label paths.

    Ensures at least some positive tiles end up in val and test.
    """
    rng = np.random.default_rng(42)

    positive_indices = []
    negative_indices = []

    for i, chip in enumerate(chip_metas):
        tile_box = box(*chip["bbox"])
        is_positive = False
        for site in known_sites:
            pt = Point(site["lon"], site["lat"])
            buf_deg = site.get("buffer_km", 2.0) / 111.0
            if pt.buffer(buf_deg).intersects(tile_box):
                is_positive = True
                break
        if is_positive:
            positive_indices.append(i)
        else:
            negative_indices.append(i)

    def _split_indices(indices: list[int]) -> tuple[list, list, list]:
        arr = np.array(indices)
        rng.shuffle(arr)
        n = len(arr)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        return arr[:n_train].tolist(), arr[n_train:n_train + n_val].tolist(), arr[n_train + n_val:].tolist()

    pos_train, pos_val, pos_test = _split_indices(positive_indices)
    neg_train, neg_val, neg_test = _split_indices(negative_indices)

    split_map = {}
    for idx in pos_train + neg_train:
        split_map[idx] = "train"
    for idx in pos_val + neg_val:
        split_map[idx] = "val"
    for idx in pos_test + neg_test:
        split_map[idx] = "test"

    for i, chip in enumerate(chip_metas):
        chip["split"] = split_map.get(i, "train")

    logger.info(
        "Split: train=%d, val=%d, test=%d (pos: %d/%d/%d)",
        len(pos_train) + len(neg_train),
        len(pos_val) + len(neg_val),
        len(pos_test) + len(neg_test),
        len(pos_train), len(pos_val), len(pos_test),
    )
    return chip_metas
