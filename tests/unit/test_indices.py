import numpy as np
import pytest

from libs.features.indices import (
    add_indices,
    build_temporal_composite,
    compute_ndvi,
    normalize,
)
from tests.fixtures.synthetic import BAND_NAMES, make_reflectance_cube, write_composite


def test_ndvi_known_values():
    nir = np.array([[0.5]], dtype=np.float32)
    red = np.array([[0.1]], dtype=np.float32)
    assert compute_ndvi(nir, red)[0, 0] == pytest.approx(0.4 / 0.6, rel=1e-5)


def test_add_indices_appends_four_channels_in_order():
    cube = make_reflectance_cube(8, 8, 4, 4, 2)
    out, names = add_indices(cube, BAND_NAMES)
    assert out.shape == (10, 8, 8)
    assert names == BAND_NAMES + ("ndvi", "ndwi", "ndbi", "bsi")
    assert np.all(np.abs(out[6:]) <= 1.0 + 1e-6)


def test_add_indices_requires_named_bands():
    with pytest.raises(ValueError, match="needs bands"):
        add_indices(np.zeros((2, 2, 2), dtype=np.float32), ("blue", "green"))


def test_composite_median_excludes_cloud_masked_pixels(tmp_path):
    a = np.full((6, 4, 4), 0.2, dtype=np.float32)
    b = np.full((6, 4, 4), 0.4, dtype=np.float32)
    c = np.full((6, 4, 4), 0.6, dtype=np.float32)
    b[:, 1, 1] = np.nan  # scene b is cloudy at (1,1)
    paths = [write_composite(tmp_path / f"s{i}.tif", x) for i, x in enumerate((a, b, c))]
    comp, profile, spec = build_temporal_composite(paths)
    assert comp[0, 0, 0] == pytest.approx(0.4)
    assert comp[0, 1, 1] == pytest.approx(0.4), "median of 0.2 and 0.6 with the cloud excluded"
    assert np.isnan(profile["nodata"])
    assert spec.names == BAND_NAMES


def test_composite_all_cloudy_pixel_is_nan(tmp_path):
    a = np.full((6, 2, 2), 0.2, dtype=np.float32)
    a[:, 0, 0] = np.nan
    paths = [write_composite(tmp_path / "s.tif", a)]
    comp, _, _ = build_temporal_composite(paths)
    assert np.isnan(comp[:, 0, 0]).all()


def test_composite_rejects_empty_and_unknown_method(tmp_path):
    with pytest.raises(ValueError, match="No scenes"):
        build_temporal_composite([])
    p = write_composite(tmp_path / "s.tif", np.zeros((6, 2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="Unknown method"):
        build_temporal_composite([p], method="max")


def test_normalize_applies_global_stats_not_per_tile():
    stats = {
        "channels": ["a", "b"],
        "p2": [0.0, 0.0],
        "p98": [1.0, 2.0],
        "mean": [0.5, 0.5],
        "std": [0.25, 0.25],
    }
    x = np.array([[[0.5]], [[1.0]]], dtype=np.float32)  # (2,1,1)
    out = normalize(x, stats)
    assert out[0, 0, 0] == pytest.approx(0.0)  # (0.5-0)/1 = 0.5 -> (0.5-0.5)/0.25
    assert out[1, 0, 0] == pytest.approx(0.0)  # (1.0-0)/2 = 0.5 -> 0
    y = np.array([[[2.0]], [[4.0]]], dtype=np.float32)  # above p98: clipped to 1 -> 2
    assert normalize(y, stats)[0, 0, 0] == pytest.approx(2.0)


def test_normalize_keeps_nan():
    stats = {"channels": ["a"], "p2": [0.0], "p98": [1.0], "mean": [0.5], "std": [0.5]}
    x = np.array([[[np.nan]]], dtype=np.float32)
    assert np.isnan(normalize(x, stats)[0, 0, 0])


def test_normalize_rejects_channel_mismatch():
    stats = {"channels": ["a"], "p2": [0.0], "p98": [1.0], "mean": [0.5], "std": [0.5]}
    with pytest.raises(ValueError, match="channels"):
        normalize(np.zeros((2, 1, 1), dtype=np.float32), stats)


def test_all_nan_composite_emits_no_warning(tmp_path):
    import warnings

    a = np.full((6, 2, 2), np.nan, dtype=np.float32)
    p = write_composite(tmp_path / "s.tif", a)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for method in ("median", "mean"):
            comp, _, _ = build_temporal_composite([p], method=method)
            assert np.isnan(comp).all()
