import numpy as np
import pytest
import rasterio

from libs.geo.bandspec import BandSpec, read_bandspec
from libs.synthetic import make_reflectance_cube, write_composite


def test_roundtrip_through_geotiff_tags(tmp_path):
    path = write_composite(tmp_path / "x.tif", make_reflectance_cube(4, 4, 2, 2, 1))
    with rasterio.open(path) as src:
        spec = read_bandspec(src)
    assert spec == BandSpec(names=("blue", "green", "red", "nir", "swir16", "swir22"))
    assert spec.n_channels == 6


def test_missing_tag_is_an_error(tmp_path):
    path = tmp_path / "plain.tif"
    with rasterio.open(
        path, "w", driver="GTiff", width=2, height=2, count=1, dtype="float32"
    ) as dst:
        dst.write(np.zeros((1, 2, 2), "float32"))
    with rasterio.open(path) as src:
        with pytest.raises(ValueError, match="BANDSPEC"):
            read_bandspec(src)


def test_with_indices():
    spec = BandSpec(names=("a", "b"), indices=("ndvi",))
    assert spec.n_channels == 3
    assert BandSpec.from_json(spec.to_json()) == spec
