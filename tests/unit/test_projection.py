import math

import pytest
from shapely.geometry import Point

from libs.geo.projection import buffer_m, distance_m, local_aeqd


def test_buffer_is_a_true_circle_at_high_latitude():
    lon, lat = 40.5777, 62.9271  # Plesetsk
    circle = buffer_m(lon, lat, 3000.0)
    # Sample the ring: every vertex must be 3000 m +- 1 m from the centre.
    for x, y in circle.exterior.coords[::16]:
        assert distance_m(lon, lat, x, y) == pytest.approx(3000.0, abs=1.0)
    # In degrees it is an ellipse: lon extent ~ 2x the lat extent at 63N
    minx, miny, maxx, maxy = circle.bounds
    assert (maxx - minx) / (maxy - miny) == pytest.approx(1 / math.cos(math.radians(lat)), rel=0.02)


def test_distance_m_matches_known_value():
    # Vandenberg SLC-4E to Cape Canaveral SLC-40 is about 3 833 km (WGS84 geodesic)
    d = distance_m(-120.6109, 34.6321, -80.5772, 28.5622)
    assert d == pytest.approx(3_833_000, rel=0.01)


def test_local_aeqd_is_centred():
    crs = local_aeqd(10.0, 50.0)
    from pyproj import Transformer

    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(10.0, 50.0)
    assert abs(x) < 1e-6 and abs(y) < 1e-6
    assert Point(x, y).equals_exact(Point(0, 0), 1e-6)
