import pytest

from libs.geo.splits import aoi_split, assign_splits, count_by_split

SITES = [
    {"name": "A", "lon": 0.0, "lat": 0.0},
    {"name": "B", "lon": 1.0, "lat": 0.0},  # ~111 km east of A
]
SPLITS = {"train": ["A"], "val": [], "test": ["B"]}


def _chip(lon, lat, tile_id):
    """Chip as chip_raster emits it: `bbox` in the raster CRS (UTM metres), plus `bbox_4326`."""
    d = 0.01
    return {
        "tile_id": tile_id,
        "bbox": [500_000.0, 4_000_000.0, 501_280.0, 4_001_280.0],
        "bbox_4326": [lon - d, lat - d, lon + d, lat + d],
    }


def test_chips_near_a_held_out_site_go_with_it():
    chips = [_chip(0.0, 0.0, "a0"), _chip(1.0, 0.05, "b0"), _chip(0.5, 0.0, "mid")]
    out = assign_splits(chips, SITES, SPLITS, exclusion_km=20.0)
    by_id = {c["tile_id"]: c["split"] for c in out}
    assert by_id["a0"] == "train"
    assert by_id["b0"] == "test"
    assert by_id["mid"] == "train", "far from every site: background chips train"


def test_chip_within_exclusion_radius_of_a_test_site_is_never_train():
    chips = [_chip(1.1, 0.0, "near_b")]  # ~11 km from B
    out = assign_splits(chips, SITES, SPLITS, exclusion_km=20.0)
    assert out[0]["split"] == "test"


def test_projected_bbox_is_not_read_as_lonlat():
    # Every chip has the same UTM `bbox`; only `bbox_4326` tells them apart.
    out = assign_splits([_chip(0.0, 0.0, "a0"), _chip(1.0, 0.0, "b0")], SITES, SPLITS)
    assert [c["split"] for c in out] == ["train", "test"]


def test_inputs_are_not_mutated():
    chips = [_chip(0.0, 0.0, "a0")]
    assign_splits(chips, SITES, SPLITS)
    assert "split" not in chips[0]


def test_aoi_split_uses_site_nearest_to_bbox_centre():
    assert aoi_split({"bbox": [-0.2, -0.2, 0.3, 0.2]}, SITES, SPLITS) == "train"
    assert aoi_split({"bbox": [0.8, -0.2, 1.3, 0.2]}, SITES, SPLITS) == "test"


def test_count_by_split():
    chips = [{"split": "train"}, {"split": "train"}, {"split": "test"}]
    assert count_by_split(chips) == {"train": 2, "test": 1}


def test_missing_site_raises():
    with pytest.raises(ValueError, match="missing"):
        assign_splits([], SITES, {"train": ["A"]})


def test_unknown_split_key_raises():
    with pytest.raises(ValueError, match="unknown split keys"):
        assign_splits([], SITES, {"train": ["A"], "vall": ["B"]})
