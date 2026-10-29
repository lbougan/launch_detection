"""Synthetic Sentinel-2-like data for the tests and the inference benchmark: small composites
with one planted 'site'.

Everything is in EPSG:32633 (UTM 33N) at 10 m so that metres are metres, with a known
geotransform, so that tests can compute where the planted site must appear.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

CRS_UTM33 = "EPSG:32633"
GSD_M = 10.0
BAND_NAMES = ("blue", "green", "red", "nir", "swir16", "swir22")
# Rough reflectance (0..1) of vegetation vs concrete/graded ground, per band.
VEGETATION = np.array([0.03, 0.06, 0.04, 0.45, 0.20, 0.10], dtype=np.float32)
CONCRETE = np.array([0.25, 0.28, 0.30, 0.35, 0.40, 0.38], dtype=np.float32)


@dataclass(frozen=True)
class PlantedSite:
    """A planted square 'launch pad' in pixel and world coordinates."""

    row: int
    col: int
    size_px: int
    x: float  # centre easting
    y: float  # centre northing


def make_reflectance_cube(
    height: int,
    width: int,
    site_row: int,
    site_col: int,
    site_size_px: int,
    seed: int = 0,
    noise: float = 0.01,
) -> np.ndarray:
    """(6, H, W) float32 reflectance: vegetation background, a concrete square, mild noise."""
    rng = np.random.default_rng(seed)
    cube = np.broadcast_to(VEGETATION[:, None, None], (6, height, width)).astype(np.float32).copy()
    r0, c0 = site_row - site_size_px // 2, site_col - site_size_px // 2
    cube[:, r0 : r0 + site_size_px, c0 : c0 + site_size_px] = CONCRETE[:, None, None]
    cube += rng.normal(0.0, noise, size=cube.shape).astype(np.float32)
    return np.clip(cube, 0.0, 1.0)


def write_composite(
    path: Path,
    cube: np.ndarray,
    origin_x: float = 500_000.0,
    origin_y: float = 4_000_000.0,
) -> Path:
    """Write a float32 GeoTIFF with NaN nodata and the BandSpec tag (see libs.geo.bandspec)."""
    from libs.geo.bandspec import BandSpec, write_bandspec

    bands, height, width = cube.shape
    transform = from_origin(origin_x, origin_y, GSD_M, GSD_M)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": width,
        "height": height,
        "count": bands,
        "crs": CRS_UTM33,
        "transform": transform,
        "nodata": float("nan"),
        "compress": "deflate",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(cube)
        write_bandspec(dst, BandSpec(names=BAND_NAMES[:bands], reflectance_scale=1.0, indices=()))
    return path


def make_synthetic_composite(
    path: Path,
    height: int = 1024,
    width: int = 1024,
    site_size_px: int = 40,
    seed: int = 0,
) -> tuple[Path, PlantedSite]:
    """Composite with one planted site near the centre. Returns (path, site)."""
    row, col = height // 2 + 37, width // 2 - 51  # off-centre so edge/centre tiles differ
    cube = make_reflectance_cube(height, width, row, col, site_size_px, seed=seed)
    origin_x, origin_y = 500_000.0, 4_000_000.0
    write_composite(path, cube, origin_x, origin_y)
    x = origin_x + (col + 0.5) * GSD_M
    y = origin_y - (row + 0.5) * GSD_M
    return path, PlantedSite(row=row, col=col, size_px=site_size_px, x=x, y=y)


def site_lonlat(site: PlantedSite) -> tuple[float, float]:
    """Planted site centre as (lon, lat) in EPSG:4326."""
    from pyproj import Transformer

    tr = Transformer.from_crs(CRS_UTM33, "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(site.x, site.y)
    return float(lon), float(lat)
