"""Chip rasters into ML-ready tiles that keep their BandSpec tag."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import Window

from libs.geo.bandspec import read_bandspec, write_bandspec

logger = logging.getLogger(__name__)

TILE_SIZE = 256


def chip_raster(
    raster_path: Path,
    output_dir: Path,
    tile_size: int = TILE_SIZE,
    overlap: int = 32,
    min_valid_frac: float = 0.5,
) -> list[dict]:
    """Slice a raster into square chips.

    Returns a list of chip metadata dicts with tile_id, path, bbox (in the raster's CRS),
    bbox_4326 (lon/lat, what site distances are computed from), row_off and col_off.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chips = []

    with rasterio.open(raster_path) as src:
        h, w = src.height, src.width
        step = tile_size - overlap
        spec = read_bandspec(src)

        for row_off in range(0, h - tile_size + 1, step):
            for col_off in range(0, w - tile_size + 1, step):
                window = Window(col_off, row_off, tile_size, tile_size)
                data = src.read(window=window)

                valid_frac = float(np.count_nonzero(~np.isnan(data[0]))) / (tile_size * tile_size)
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
                    write_bandspec(dst, spec)

                chips.append(
                    {
                        "tile_id": tile_id,
                        "path": str(out_path),
                        "bbox": list(bounds),
                        "bbox_4326": list(transform_bounds(src.crs, "EPSG:4326", *bounds)),
                        "row_off": row_off,
                        "col_off": col_off,
                    }
                )

    logger.info(
        "Chipped %s into %d tiles (%dpx, overlap=%d)", raster_path, len(chips), tile_size, overlap
    )
    return chips


def load_known_sites(manifest_path: Path) -> list[dict]:
    """Load known launch sites from the JSON manifest."""
    with open(manifest_path) as f:
        data = json.load(f)
    return data["sites"]
