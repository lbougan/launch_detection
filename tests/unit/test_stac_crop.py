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


def test_bbox_to_native_is_densified_at_high_latitude():
    from pyproj import Transformer
    from rasterio.crs import CRS

    from libs.stac.client import _bbox_to_native

    bbox = [24.0, 69.0, 30.0, 71.0]  # northern Norway/Finland, UTM 35N
    crs = CRS.from_epsg(32635)
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    w2, s2 = tr.transform(bbox[0], bbox[1])
    e2, n2 = tr.transform(bbox[2], bbox[3])
    w, s, e, n = _bbox_to_native(bbox, crs)
    assert w <= w2 and s <= s2 and e >= e2 and n >= n2
    assert (e - w) > (e2 - w2) + 1_000, "the two-corner window misses the bulging edges"


def test_bbox_to_native_is_identity_in_epsg4326():
    from rasterio.crs import CRS

    from libs.stac.client import _bbox_to_native

    out = _bbox_to_native([10.0, 20.0, 11.0, 21.0], CRS.from_epsg(4326))
    assert out == pytest.approx((10.0, 20.0, 11.0, 21.0))


def test_aoi_target_grid_is_one_utm_grid_at_10_m():
    from libs.stac.client import aoi_target_grid

    g = aoi_target_grid([80.1, 13.6, 80.4, 13.9])  # Sriharikota, UTM 44N
    assert g.crs.to_epsg() == 32644
    assert g.transform.a == 10.0 and g.transform.e == -10.0
    assert 3150 < g.width < 3400 and 3250 < g.height < 3400, (g.width, g.height)
    assert g.transform.c % 10 == 0 and g.transform.f % 10 == 0, "snapped to the 10 m lattice"
    south = aoi_target_grid([-52.9, 5.1, -52.6, 5.4])  # Kourou, UTM 22N
    assert south.crs.to_epsg() == 32622
    assert aoi_target_grid([-52.9, -5.4, -52.6, -5.1]).crs.to_epsg() == 32722


def _write_band(path, data, crs, transform, nodata=None):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[1],
        height=data.shape[0],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)
    return str(path)


def test_download_scene_crop_lands_every_scene_on_the_aoi_grid(tmp_path):
    """Two 'scenes' in different UTM zones, read as local files, end up on one identical grid."""
    from rasterio.transform import from_origin

    from libs.stac.client import SceneMetadata, aoi_target_grid, download_scene_crop

    bbox = [17.95, 40.0, 18.05, 40.05]  # straddles the 33N / 34N boundary at 18 E
    grid = aoi_target_grid(bbox)
    out = []
    for i, epsg in enumerate((32633, 32634)):
        tr = rasterio.warp.transform("EPSG:4326", f"EPSG:{epsg}", [bbox[0]], [bbox[3]])
        x0, y0 = tr[0][0] - 2_000, tr[1][0] + 2_000
        # the scene covers only the western 60 % of the AOI width (about 5 km of 8.5 km)
        transform = from_origin(x0, y0, 10.0, 10.0)
        h, w = 1000, 700
        assets = {
            b: _write_band(
                tmp_path / f"{i}_{b}.tif",
                np.full((h, w), 2000, dtype=np.uint16),
                f"EPSG:{epsg}",
                transform,
                nodata=0,
            )
            for b in S2_BANDS
        }
        assets["SCL"] = _write_band(
            tmp_path / f"{i}_SCL.tif",
            np.full((h // 2, w // 2), 4, dtype=np.uint8),
            f"EPSG:{epsg}",
            from_origin(x0, y0, 20.0, 20.0),
            nodata=0,
        )
        scene = SceneMetadata(f"s{i}", "sentinel-2", "", bbox, 0.0, assets)
        out.append(download_scene_crop(scene, grid, tmp_path / "crops"))
    for p in out:
        with rasterio.open(p) as src:
            assert src.crs == grid.crs and src.transform == grid.transform
            assert (src.height, src.width) == (grid.height, grid.width)
            cube = src.read()
        assert np.allclose(cube[:, 10, 10], 0.2), "inside the scene: reflectance"
        assert np.isnan(cube[:, 10, -10]).all(), "outside the scene: NaN"
