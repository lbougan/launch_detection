"""Cloud masking and SCL-based filtering for Sentinel-2."""

from __future__ import annotations

import numpy as np

SCL_CLEAR_CLASSES = {4, 5, 6, 7}  # vegetation, bare, water, unclassified-clear


def scl_cloud_mask(scl_band: np.ndarray) -> np.ndarray:
    """Return a boolean mask where True = clear pixel (usable).

    The SCL (Scene Classification Layer) band values:
      0: no data, 1: saturated, 2: dark/shadow, 3: cloud shadow,
      4: vegetation, 5: bare soil, 6: water, 7: unclassified,
      8: cloud medium, 9: cloud high, 10: cirrus, 11: snow/ice
    """
    return np.isin(scl_band, list(SCL_CLEAR_CLASSES))


def apply_cloud_mask(
    bands: np.ndarray,
    scl_band: np.ndarray,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Mask out cloudy/bad pixels by setting them to fill_value.

    Args:
        bands: (C, H, W) multi-band array.
        scl_band: (H, W) SCL classification.
        fill_value: Value to assign masked pixels.
    """
    clear = scl_cloud_mask(scl_band)
    masked = bands.copy()
    masked[:, ~clear] = fill_value
    return masked
