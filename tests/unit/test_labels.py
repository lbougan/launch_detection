import numpy as np
import rasterio

from libs.geo.labels import (
    IGNORE,
    NEGATIVE,
    POSITIVE,
    build_label_mask,
    read_label,
    weight_from_label,
    write_label,
)
from tests.fixtures.synthetic import site_lonlat


def test_label_values_are_positive_ignore_ring_negative(small_composite):
    path, site = small_composite
    lon, lat = site_lonlat(site)
    with rasterio.open(path) as src:
        mask = build_label_mask(src, [{"lon": lon, "lat": lat, "buffer_km": 0.3}])
    assert mask.shape == (320, 320)
    assert mask[site.row, site.col] == POSITIVE
    # 450 m from the centre = inside the ignore ring [300 m, 600 m)
    assert mask[site.row, site.col + 45] == IGNORE
    # 700 m away = negative
    assert mask[site.row, site.col + 70] == NEGATIVE
    assert set(np.unique(mask).tolist()) <= {NEGATIVE, IGNORE, POSITIVE}


def test_positive_radius_in_pixels_matches_metres(small_composite):
    path, site = small_composite
    lon, lat = site_lonlat(site)
    with rasterio.open(path) as src:
        mask = build_label_mask(src, [{"lon": lon, "lat": lat, "buffer_km": 0.5}])
    row = mask[site.row]
    positives = np.flatnonzero(row == POSITIVE)
    radius_px = (positives.max() - positives.min() + 1) / 2
    assert abs(radius_px - 50) <= 2  # 500 m at 10 m


def test_weight_zero_on_ignore_ring():
    mask = np.array([[NEGATIVE, IGNORE, POSITIVE]], dtype=np.float32)
    assert weight_from_label(mask).tolist() == [[1.0, 0.0, 1.0]]


def test_label_roundtrip_geotiff(tmp_path, small_composite):
    path, site = small_composite
    with rasterio.open(path) as src:
        lon, lat = site_lonlat(site)
        mask = build_label_mask(src, [{"lon": lon, "lat": lat, "buffer_km": 0.3}])
        out = write_label(tmp_path / "label.tif", mask, src)
    assert read_label(out).tolist() == mask.tolist()
