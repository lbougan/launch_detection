import numpy as np
import rasterio

from libs.geo.bandspec import read_bandspec
from tests.fixtures.synthetic import BAND_NAMES, site_lonlat


def test_small_composite_has_bandspec_and_planted_site(small_composite):
    path, site = small_composite
    with rasterio.open(path) as src:
        spec = read_bandspec(src)
        cube = src.read()
        assert src.crs.to_epsg() == 32633
    assert spec.names == BAND_NAMES
    assert cube.shape == (6, 320, 320)
    # concrete is brighter than vegetation in the red band
    assert cube[2, site.row, site.col] > 0.2
    assert cube[2, 5, 5] < 0.1
    lon, lat = site_lonlat(site)
    assert 14.0 < lon < 16.0 and 35.0 < lat < 37.0  # UTM 33N near 500000E, 4000000N


def test_site_world_coordinates_match_pixels(small_composite):
    path, site = small_composite
    with rasterio.open(path) as src:
        row, col = src.index(site.x, site.y)
        assert np.isnan(src.nodata)
    assert (row, col) == (site.row, site.col)
