"""Spectral indices, temporal compositing and the single normalisation step."""

from __future__ import annotations

import logging
import warnings
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from libs.geo.bandspec import BandSpec, read_bandspec

logger = logging.getLogger(__name__)

INDEX_NAMES = ("ndvi", "ndwi", "ndbi", "bsi")


def _safe_ratio(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (a - b) / (a + b + eps)


def compute_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    return _safe_ratio(nir, red)


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return _safe_ratio(green, nir)


def compute_ndbi(swir: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Normalised Difference Built-up Index: built surfaces are bright in SWIR."""
    return _safe_ratio(swir, nir)


def compute_bsi(blue: np.ndarray, red: np.ndarray, nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Bare Soil Index: graded or cleared ground."""
    num = (swir + red) - (nir + blue)
    den = (swir + red) + (nir + blue) + 1e-8
    return num / den


def add_indices(stack: np.ndarray, names: tuple[str, ...]) -> tuple[np.ndarray, tuple[str, ...]]:
    """Append NDVI, NDWI, NDBI, BSI computed from *reflectance* bands.

    `names` must contain blue, green, red, nir, swir16. Returns (stack+4, names+4).
    """
    idx = {n: i for i, n in enumerate(names)}
    missing = [n for n in ("blue", "green", "red", "nir", "swir16") if n not in idx]
    if missing:
        raise ValueError(f"add_indices needs bands {missing}")
    b = stack.astype(np.float32)
    blue, green, red, nir, swir16 = (b[idx[n]] for n in ("blue", "green", "red", "nir", "swir16"))
    extra = np.stack(
        [
            compute_ndvi(nir, red),
            compute_ndwi(green, nir),
            compute_ndbi(swir16, nir),
            compute_bsi(blue, red, nir, swir16),
        ]
    )
    return np.concatenate([b, extra], axis=0), names + INDEX_NAMES


COMPOSITE_BLOCK_ROWS = 512  # rows per block: peak RAM is one block x scenes x bands


def _grid_of(src) -> tuple:
    return (src.crs, src.transform, src.width, src.height, src.count)


def build_temporal_composite(
    scene_paths: list[Path], method: str = "median", block_rows: int = COMPOSITE_BLOCK_ROWS
) -> tuple[np.ndarray, dict, BandSpec]:
    """NaN-aware pixel-wise median (or mean) of reflectance scenes on one identical grid.

    Cloud-masked pixels are NaN in the inputs (see libs.geo.masks) and are therefore ignored;
    a pixel that is cloudy in every scene stays NaN. Every scene must share CRS, transform and
    shape (ingest warps them onto one AOI grid); the reduction runs in blocks of `block_rows`
    rows. Returns (composite, profile, bandspec).
    """
    if not scene_paths:
        raise ValueError("No scenes provided for compositing")
    if method not in ("median", "mean"):
        raise ValueError(f"Unknown method: {method}")
    with rasterio.open(scene_paths[0]) as src:
        profile = src.profile.copy()
        spec = read_bandspec(src)
        ref = _grid_of(src)
    for p in scene_paths[1:]:
        with rasterio.open(p) as src:
            if _grid_of(src) != ref:
                raise ValueError(
                    f"{p} is not on the grid of {scene_paths[0]}: "
                    f"{_grid_of(src)} != {ref} (crs, transform, width, height, bands)"
                )
    _, _, width, height, count = ref
    reducer = np.nanmedian if method == "median" else np.nanmean
    composite = np.empty((count, height, width), dtype=np.float32)
    with ExitStack() as stack:
        sources = [stack.enter_context(rasterio.open(p)) for p in scene_paths]
        for r0 in range(0, height, block_rows):
            window = Window(0, r0, width, min(block_rows, height - r0))
            block = np.stack([s.read(window=window).astype(np.float32) for s in sources])
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="All-NaN slice", category=RuntimeWarning)
                warnings.filterwarnings(
                    "ignore", message="Mean of empty slice", category=RuntimeWarning
                )
                composite[:, r0 : r0 + window.height] = reducer(block, axis=0)
    profile.update(dtype="float32", nodata=float("nan"), count=count)
    return composite, profile, spec


def check_stats_channels(names: tuple[str, ...], stats: dict) -> None:
    """Raise unless `names` (bands + indices, in order) equal the channels the stats describe."""
    if tuple(names) != tuple(stats["channels"]):
        raise ValueError(
            f"input channels {list(names)} do not match the stats channels {stats['channels']}"
        )


def normalize(array: np.ndarray, stats: dict) -> np.ndarray:
    """The one normalisation used at train and serve time (decision 4).

    Per channel c: clip((x - p2[c]) / (p98[c] - p2[c]), 0, 1), then (x - mean[c]) / std[c].
    NaN stays NaN. `array` is (C,H,W) and C must equal len(stats["channels"]).
    """
    n = len(stats["channels"])
    if array.shape[0] != n:
        raise ValueError(f"array has {array.shape[0]} channels, stats has {n} channels")
    p2 = np.asarray(stats["p2"], dtype=np.float32)[:, None, None]
    p98 = np.asarray(stats["p98"], dtype=np.float32)[:, None, None]
    mean = np.asarray(stats["mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(stats["std"], dtype=np.float32)[:, None, None]
    x = array.astype(np.float32)
    x = np.clip((x - p2) / np.maximum(p98 - p2, 1e-6), 0.0, 1.0)
    return (x - mean) / np.maximum(std, 1e-6)
