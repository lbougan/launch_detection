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


def test_three_km_buffer_at_63_north_is_round(tmp_path):
    from pyproj import Transformer
    from rasterio.transform import from_origin

    lon, lat = 40.5777, 62.9271  # Plesetsk
    x, y = Transformer.from_crs("EPSG:4326", "EPSG:32637", always_xy=True).transform(lon, lat)
    half = 700  # px: 7 km each way at 10 m
    path = tmp_path / "grid.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2 * half,
        height=2 * half,
        count=1,
        dtype="uint8",
        crs="EPSG:32637",
        transform=from_origin(x - half * 10.0, y + half * 10.0, 10.0, 10.0),
    ) as dst:
        dst.write(np.zeros((2 * half, 2 * half), dtype=np.uint8), 1)
    with rasterio.open(path) as src:
        mask = build_label_mask(src, [{"lon": lon, "lat": lat, "buffer_km": 3.0}])
    pos = mask == POSITIVE
    rows = np.flatnonzero(pos.any(axis=1))
    cols = np.flatnonzero(pos.any(axis=0))
    radius_rows = (rows[-1] - rows[0] + 1) / 2
    radius_cols = (cols[-1] - cols[0] + 1) / 2
    assert abs(radius_rows - 300) <= 2 and abs(radius_cols - 300) <= 2, (radius_rows, radius_cols)
