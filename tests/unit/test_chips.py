import rasterio

from libs.geo.bandspec import read_bandspec
from libs.geo.chips import chip_raster


def test_chips_cover_the_raster_and_carry_bandspec(small_composite, tmp_path):
    path, _ = small_composite
    chips = chip_raster(path, tmp_path / "tiles", tile_size=128, overlap=32)
    # 320 px, step 96: rows at 0, 96, 192 -> 3x3 = 9 chips (edge handled in inference, not chips)
    assert len(chips) == 9
    with rasterio.open(chips[0]["path"]) as src:
        assert src.count == 6 and src.width == 128
        assert read_bandspec(src).names[0] == "blue"
    assert chips[0]["bbox"][0] < chips[0]["bbox"][2]
    assert all("tile_id" in c and "bbox" in c for c in chips)
