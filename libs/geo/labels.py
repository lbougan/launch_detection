"""Weak labels from known-site coordinates (decision 1).

POSITIVE (1.0) inside `buffer_km`, IGNORE (0.5) in the ring [buffer_km, 2 x buffer_km),
NEGATIVE (0.0) beyond. IGNORE pixels get loss weight 0: the circle is a coarse proxy for an
irregular footprint, and the ring is where the label is least trustworthy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from shapely.geometry import mapping, shape

from libs.geo.projection import buffer_m

NEGATIVE, IGNORE, POSITIVE = 0.0, 0.5, 1.0
IGNORE_RING_FACTOR = 2.0
DEFAULT_BUFFER_KM = 2.0


def _site_shapes_in_raster_crs(src: rasterio.io.DatasetReader, sites: list[dict], factor: float):
    shapes = []
    for s in sites:
        radius_m = float(s.get("buffer_km", DEFAULT_BUFFER_KM)) * 1000.0 * factor
        circle_ll = buffer_m(float(s["lon"]), float(s["lat"]), radius_m)
        geom = transform_geom("EPSG:4326", src.crs, mapping(circle_ll))
        shapes.append(shape(geom))
    return shapes


def build_label_mask(src: rasterio.io.DatasetReader, sites: list[dict]) -> np.ndarray:
    """(H, W) float32 mask in {0, 0.5, 1} for the raster `src` (any CRS)."""
    out_shape = (src.height, src.width)
    mask = np.full(out_shape, NEGATIVE, dtype=np.float32)
    ring = _site_shapes_in_raster_crs(src, sites, IGNORE_RING_FACTOR)
    core = _site_shapes_in_raster_crs(src, sites, 1.0)
    if ring:
        mask = rasterize(
            [(mapping(g), IGNORE) for g in ring],
            out_shape=out_shape,
            transform=src.transform,
            fill=NEGATIVE,
            dtype="float32",
        )
    if core:
        pos = rasterize(
            [(mapping(g), 1) for g in core],
            out_shape=out_shape,
            transform=src.transform,
            fill=0,
            dtype="uint8",
        )
        mask[pos == 1] = POSITIVE
    return mask


def weight_from_label(mask: np.ndarray) -> np.ndarray:
    """Loss weight: 0 on the ignore ring, 1 elsewhere."""
    return np.where(mask == IGNORE, 0.0, 1.0).astype(np.float32)


def write_label(path: Path, mask: np.ndarray, like: rasterio.io.DatasetReader) -> Path:
    """Georeferenced single-band label next to its chip."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = like.profile.copy()
    profile.update(count=1, dtype="float32", nodata=None, compress="deflate")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mask.astype(np.float32), 1)
    return path


def read_label(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)
