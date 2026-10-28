import numpy as np
import pytest
import rasterio

from libs.stac.client import (
    S2_BAND_NAMES,
    S2_BANDS,
    assemble_crop,
    is_pipeline_crop,
    scale_offset_for,
)
from tests.fixtures.synthetic import write_composite


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


def test_scale_offset_from_processing_baseline():
    assert scale_offset_for({}, {"s2:processing_baseline": "05.10"}) == (1e-4, -0.1)
    assert scale_offset_for({}, {"s2:processing_baseline": "03.00"}) == (1e-4, 0.0)
    assert scale_offset_for({}, {}) == (1e-4, 0.0)


def test_explicit_raster_bands_wins():
    extra = {"raster:bands": [{"scale": 2e-4, "offset": 0.5}]}
    assert scale_offset_for(extra, {"s2:processing_baseline": "05.10"}) == (2e-4, 0.5)


def test_is_pipeline_crop(tmp_path):
    tagged = write_composite(tmp_path / "t.tif", np.zeros((6, 2, 2), dtype=np.float32))
    plain = tmp_path / "p.tif"
    with rasterio.open(plain, "w", driver="GTiff", width=2, height=2, count=1, dtype="uint16") as d:
        d.write(np.zeros((1, 2, 2), "uint16"))
    assert is_pipeline_crop(tagged)
    assert not is_pipeline_crop(plain)
