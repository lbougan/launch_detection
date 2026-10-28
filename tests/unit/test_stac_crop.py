import numpy as np
import pytest

from libs.stac.client import S2_BAND_NAMES, S2_BANDS, assemble_crop


def _arrays(dn: int) -> dict[str, np.ndarray]:
    return {b: np.full((2, 2), dn, dtype=np.uint16) for b in S2_BANDS}


def test_dn_converted_to_reflectance_in_band_order():
    scales = {b: (1e-4, 0.0) for b in S2_BANDS}
    scl = np.full((2, 2), 4, dtype=np.uint8)
    out = assemble_crop(_arrays(2000), scl, scales)
    assert out.shape == (len(S2_BAND_NAMES), 2, 2)
    assert out.dtype == np.float32
    assert np.allclose(out, 0.2)


def test_offset_applied():
    scales = {b: (1e-4, -0.1) for b in S2_BANDS}
    scl = np.full((2, 2), 4, dtype=np.uint8)
    assert np.allclose(assemble_crop(_arrays(2000), scl, scales), 0.1)


def test_missing_scale_falls_back_and_cloud_is_nan_in_every_band():
    scl = np.array([[4, 9], [4, 4]], dtype=np.uint8)
    out = assemble_crop(_arrays(2000), scl, {})
    assert np.isnan(out[:, 0, 1]).all()
    assert np.allclose(out[:, 0, 0], 0.2)


def test_missing_band_raises():
    arrays = _arrays(1)
    del arrays["B08"]
    with pytest.raises(ValueError, match="B08"):
        assemble_crop(arrays, np.full((2, 2), 4, dtype=np.uint8), {})
