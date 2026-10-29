"""Cloud masking from the Sentinel-2 Scene Classification Layer (SCL)."""

from __future__ import annotations

import numpy as np

# 4 vegetation, 5 bare soil, 6 water, 7 unclassified; everything else is cloud, shadow, snow or no-data
SCL_CLEAR_CLASSES = (4, 5, 6, 7)


def scl_cloud_mask(scl_band: np.ndarray) -> np.ndarray:
    """True where the pixel is usable."""
    return np.isin(scl_band, SCL_CLEAR_CLASSES)


def apply_cloud_mask(bands: np.ndarray, scl_band: np.ndarray) -> np.ndarray:
    """Return a copy of `bands` (C,H,W) with non-clear pixels set to NaN."""
    clear = scl_cloud_mask(scl_band)
    out = bands.astype(np.float32, copy=True)
    out[:, ~clear] = np.nan
    return out
