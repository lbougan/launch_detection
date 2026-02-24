"""Spectral indices and composite generation for Sentinel-2 imagery."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

logger = logging.getLogger(__name__)

S2_BAND_ORDER = {"blue": 0, "green": 1, "red": 2, "nir": 3, "swir16": 4, "swir22": 5}


def _safe_ratio(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (a - b) / (a + b + eps)


def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    return _safe_ratio(nir, red)


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return _safe_ratio(green, nir)


def compute_ndbi(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Normalized Difference Built-up Index — highlights built surfaces."""
    return _safe_ratio(swir, nir)


def compute_bsi(blue: np.ndarray, red: np.ndarray, nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Bare Soil Index — useful for cleared / graded areas."""
    num = (swir + red) - (nir + blue)
    den = (swir + red) + (nir + blue) + 1e-8
    return num / den


def add_indices(stack: np.ndarray) -> np.ndarray:
    """Append NDVI, NDWI, NDBI, BSI bands to a 6-band Sentinel-2 stack.

    Input shape:  (6, H, W) — blue, green, red, nir, swir16, swir22
    Output shape: (10, H, W)
    """
    blue, green, red, nir, swir16, swir22 = [stack[i].astype(np.float32) for i in range(6)]
    ndvi = compute_ndvi(nir, red)
    ndwi = compute_ndwi(green, nir)
    ndbi = compute_ndbi(swir16, nir)
    bsi = compute_bsi(blue, red, nir, swir16)
    return np.concatenate([stack.astype(np.float32), ndvi[None], ndwi[None], ndbi[None], bsi[None]], axis=0)


def build_temporal_composite(
    scene_paths: list[Path],
    method: str = "median",
) -> tuple[np.ndarray, dict]:
    """Build a temporal composite from multiple scenes.

    Args:
        scene_paths: Paths to multi-band GeoTIFFs (same extent/resolution).
        method: "median" or "mean".

    Returns:
        (composite_array, rasterio_profile)
    """
    arrays = []
    profile = None
    for p in scene_paths:
        with rasterio.open(p) as src:
            arrays.append(src.read())
            if profile is None:
                profile = src.profile.copy()

    if not arrays:
        raise ValueError("No scenes provided for compositing")

    cube = np.stack(arrays, axis=0)  # (T, C, H, W)
    if method == "median":
        composite = np.nanmedian(cube, axis=0)
    elif method == "mean":
        composite = np.nanmean(cube, axis=0)
    else:
        raise ValueError(f"Unknown method: {method}")

    return composite.astype(np.float32), profile


def normalize_percentile(
    array: np.ndarray,
    low: float = 2.0,
    high: float = 98.0,
) -> np.ndarray:
    """Per-band percentile normalization to [0, 1]."""
    result = np.empty_like(array, dtype=np.float32)
    for i in range(array.shape[0]):
        band = array[i]
        lo = np.nanpercentile(band, low)
        hi = np.nanpercentile(band, high)
        if hi - lo < 1e-8:
            result[i] = 0.0
        else:
            result[i] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
    return result
