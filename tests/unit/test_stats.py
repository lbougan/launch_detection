import json

import numpy as np
import pytest

from libs.features.stats import compute_channel_stats, load_stats, save_stats
from tests.fixtures.synthetic import BAND_NAMES, make_reflectance_cube, write_composite


def test_stats_over_training_composites(tmp_path):
    cubes = [make_reflectance_cube(64, 64, 32, 32, 8, seed=s) for s in range(2)]
    paths = [write_composite(tmp_path / f"c{i}.tif", c) for i, c in enumerate(cubes)]
    stats = compute_channel_stats(paths, sample_px=4096)
    assert stats["channels"] == list(BAND_NAMES + ("ndvi", "ndwi", "ndbi", "bsi"))
    assert len(stats["p2"]) == 10 and len(stats["std"]) == 10
    assert all(0.0 < s for s in stats["std"])
    assert all(lo < hi for lo, hi in zip(stats["p2"], stats["p98"], strict=True))
    out = tmp_path / "stats.json"
    save_stats(stats, out)
    assert load_stats(out) == json.loads(out.read_text())


def test_stats_deterministic_and_reject_all_nan(tmp_path):
    cube = make_reflectance_cube(64, 64, 32, 32, 8)
    p = write_composite(tmp_path / "c.tif", cube)
    assert compute_channel_stats([p], sample_px=500) == compute_channel_stats([p], sample_px=500)
    nan = write_composite(tmp_path / "n.tif", np.full((6, 4, 4), np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match="no valid pixels"):
        compute_channel_stats([nan])


def test_stats_reject_mismatched_channel_names(tmp_path):
    a = write_composite(tmp_path / "a.tif", make_reflectance_cube(16, 16, 8, 8, 2))
    b = write_composite(tmp_path / "b.tif", make_reflectance_cube(16, 16, 8, 8, 2))
    import rasterio

    from libs.geo.bandspec import BandSpec, write_bandspec

    with rasterio.open(b, "r+") as dst:
        write_bandspec(dst, BandSpec(names=("green", "blue", "red", "nir", "swir16", "swir22")))
    with pytest.raises(ValueError, match="channel"):
        compute_channel_stats([a, b])
