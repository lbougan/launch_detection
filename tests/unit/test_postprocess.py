import dataclasses

import numpy as np
import rasterio
from rasterio.transform import from_origin

from libs.geo.postprocess import (
    Detection,
    deduplicate_detections,
    score_detections,
    threshold_and_extract,
)


def _prob_raster(tmp_path, blob_value=0.9):
    prob = np.zeros((64, 64), dtype=np.float32)
    prob[20:30, 20:30] = blob_value  # 10x10 px = 1 ha at 10 m
    path = tmp_path / "prob.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=64,
        height=64,
        count=1,
        dtype="float32",
        crs="EPSG:32633",
        transform=from_origin(500_000, 4_000_000, 10, 10),
        nodata=float("nan"),
    ) as dst:
        dst.write(prob, 1)
    return path


def test_extract_one_blob_with_area_in_km2_and_lonlat_centroid(tmp_path):
    dets = threshold_and_extract(_prob_raster(tmp_path), threshold=0.5, min_area_px=10)
    assert len(dets) == 1
    d = dets[0]
    assert abs(d.prob - 0.9) < 1e-6
    assert d.area_km2 == 0.01
    assert 14 < d.centroid_lon < 16 and 35 < d.centroid_lat < 37, "centroid is in EPSG:4326"
    assert d.rank_score is None


def test_score_detections_is_immutable_and_keeps_prob():
    d = Detection(
        geometry=None,
        prob=0.8,
        rank_score=None,
        area_km2=2.0,
        compactness=0.9,
        centroid_lon=0,
        centroid_lat=0,
        evidence={},
    )
    out = score_detections([d])
    assert d.rank_score is None, "input untouched"
    assert out[0].prob == 0.8
    assert out[0].rank_score is not None and 0 < out[0].rank_score <= 1
    assert dataclasses.is_dataclass(out[0]) and out[0].__dataclass_params__.frozen


def test_dedupe_uses_metres():
    a = Detection(
        geometry=None,
        prob=0.9,
        rank_score=0.9,
        area_km2=1,
        compactness=1,
        centroid_lon=40.5777,
        centroid_lat=62.9271,
        evidence={},
    )
    b = dataclasses.replace(a, prob=0.8, rank_score=0.8, centroid_lon=40.5777 + 0.03)
    c = dataclasses.replace(a, prob=0.7, rank_score=0.7, centroid_lat=62.9271 + 0.03)
    out = deduplicate_detections([a, b, c], eps_m=2000.0)
    assert [d.prob for d in out] == [0.9, 0.7]


def test_extracted_polygon_is_lonlat_and_matches_the_centroid(tmp_path):
    """Detection.geometry is EPSG:4326 (the store and the API stamp SRID 4326 on it)."""
    d = threshold_and_extract(_prob_raster(tmp_path), threshold=0.5, min_area_px=10)[0]
    minx, miny, maxx, maxy = d.geometry.bounds
    assert -180 <= minx <= maxx <= 180 and -90 <= miny <= maxy <= 90
    c = d.geometry.centroid
    assert abs(c.x - d.centroid_lon) < 1e-4 and abs(c.y - d.centroid_lat) < 1e-4
    assert maxx - minx < 0.01, "a 100 m blob spans well under 0.01 deg"
    assert d.area_km2 == 0.01, "area comes from projected pixels, not degrees"
    assert 0.7 < d.compactness < 0.8, "square Polsby-Popper (pi/4) on the projected polygon"


def test_each_component_keeps_its_own_probability_and_area(tmp_path):
    prob = np.zeros((64, 64), dtype=np.float32)
    prob[5:15, 5:15] = 0.9
    prob[40:60, 40:50] = 0.6
    prob[30, 30] = 0.99  # 1 px: below min_area
    path = tmp_path / "two.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=64,
        height=64,
        count=1,
        dtype="float32",
        crs="EPSG:32633",
        transform=from_origin(500_000, 4_000_000, 10, 10),
    ) as dst:
        dst.write(prob, 1)
    dets = threshold_and_extract(path, threshold=0.5, min_area_px=10)
    assert [round(d.prob, 6) for d in dets] == [0.9, 0.6]
    assert [d.evidence["area_px"] for d in dets] == [100, 200]
