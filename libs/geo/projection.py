"""Metric geometry helpers: a local azimuthal equidistant CRS per point."""

from __future__ import annotations

from pyproj import CRS, Geod, Transformer
from shapely.geometry import Point, Polygon
from shapely.ops import transform as shp_transform

_GEOD = Geod(ellps="WGS84")


def local_aeqd(lon: float, lat: float) -> CRS:
    """Azimuthal equidistant CRS centred on (lon, lat): metres, true distances from the centre."""
    return CRS.from_proj4(f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs")


def buffer_m(lon: float, lat: float, radius_m: float, resolution: int = 64) -> Polygon:
    """A circle of `radius_m` metres around (lon, lat), returned in EPSG:4326."""
    crs = local_aeqd(lon, lat)
    to_ll = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform
    circle = Point(0.0, 0.0).buffer(radius_m, quad_segs=resolution)
    return shp_transform(to_ll, circle)


def distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Geodesic distance in metres."""
    _, _, d = _GEOD.inv(lon1, lat1, lon2, lat2)
    return float(d)
