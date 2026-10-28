import numpy as np
import pytest
import rasterio

from libs.inference.predictors import FakePredictor
from libs.inference.sliding_window import Accumulator, iter_windows, run_sliding_window
from tests.fixtures.synthetic import make_reflectance_cube, write_composite


@pytest.mark.parametrize("h,w", [(256, 256), (300, 300), (1000, 700)])
def test_windows_cover_every_pixel(h, w):
    windows = list(iter_windows(h, w, tile=256, overlap=64))
    count = np.zeros((h, w), dtype=np.int32)
    for r, c in windows:
        count[r : r + 256, c : c + 256] += 1
    assert count.min() >= 1
    # windows never start outside the raster, and the last one ends exactly at the edge
    assert all(0 <= r <= h - 256 and 0 <= c <= w - 256 for r, c in windows)
    assert max(r for r, _ in windows) == h - 256
    assert max(c for _, c in windows) == w - 256


def test_small_raster_is_padded_to_one_window():
    windows = list(iter_windows(100, 120, tile=256, overlap=64))
    assert windows == [(0, 0)]


def test_overlap_must_be_smaller_than_tile():
    with pytest.raises(ValueError):
        list(iter_windows(100, 100, tile=64, overlap=64))


def test_accumulator_memmap_equals_in_memory(tmp_path):
    rng = np.random.default_rng(0)
    tiles = [
        ((0, 0), rng.random((8, 8), dtype=np.float32)),
        ((4, 4), rng.random((8, 8), dtype=np.float32)),
    ]
    a = Accumulator((12, 12), memmap=False)
    b = Accumulator((12, 12), memmap=True, tmpdir=tmp_path)
    for (r, c), t in tiles:
        a.add(r, c, t)
        b.add(r, c, t)
    np.testing.assert_allclose(a.finalize(), b.finalize(), atol=1e-6)
    assert np.isnan(a.finalize()[11, 0]), "untouched pixels are NaN, not 0"
    b.close()
    assert list(tmp_path.iterdir()) == [], "close() removes the temp dir"


def test_run_sliding_window_writes_prob_raster_with_edge_coverage(tmp_path):
    cube = make_reflectance_cube(300, 290, 150, 145, 20)
    src_path = write_composite(tmp_path / "c.tif", cube)
    stats = _stats_for(src_path)
    out = run_sliding_window(
        raster_path=src_path,
        output_path=tmp_path / "prob.tif",
        predictor=FakePredictor(value=0.7),
        stats=stats,
        tile=128,
        overlap=32,
        batch_size=4,
        memmap=False,
    )
    with rasterio.open(out) as src:
        prob = src.read(1)
        assert src.width == 290 and src.height == 300
        assert src.crs.to_epsg() == 32633
        assert src.count == 1 and np.isnan(src.nodata)
        assert "BANDSPEC" not in src.tags()
    assert not np.isnan(prob).any(), "every pixel incl. the right/bottom strip is predicted"
    np.testing.assert_allclose(prob, 0.7, atol=1e-6)


def test_run_sliding_window_small_raster_and_memmap(tmp_path):
    cube = make_reflectance_cube(100, 120, 50, 60, 10)
    src_path = write_composite(tmp_path / "s.tif", cube)
    out = run_sliding_window(
        src_path,
        tmp_path / "p.tif",
        FakePredictor(0.4),
        _stats_for(src_path),
        tile=256,
        overlap=64,
        memmap=True,
        tmpdir=tmp_path,
    )
    with rasterio.open(out) as src:
        assert (src.height, src.width) == (100, 120)
        np.testing.assert_allclose(src.read(1), 0.4, atol=1e-6)


def _stats_for(path):
    from libs.features.stats import compute_channel_stats

    return compute_channel_stats([path], sample_px=2000)
