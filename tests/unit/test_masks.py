import numpy as np

from libs.geo.masks import apply_cloud_mask, scl_cloud_mask


def test_scl_clear_classes():
    scl = np.array([[0, 4, 5], [6, 7, 9], [8, 10, 11]], dtype=np.uint8)
    clear = scl_cloud_mask(scl)
    assert clear.tolist() == [[False, True, True], [True, True, False], [False, False, False]]


def test_apply_cloud_mask_fills_nan_not_zero():
    bands = np.ones((2, 2, 2), dtype=np.float32)
    scl = np.array([[4, 9], [4, 4]], dtype=np.uint8)
    out = apply_cloud_mask(bands, scl)
    assert np.isnan(out[:, 0, 1]).all()
    assert out[:, 0, 0].tolist() == [1.0, 1.0]
    assert bands[0, 0, 1] == 1.0, "input must not be mutated"
