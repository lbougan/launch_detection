"""Train/serve parity: the dataset and the sliding-window reader preprocess identically."""

import numpy as np
import rasterio

from libs.features.stats import compute_channel_stats, save_stats
from libs.geo.bandspec import read_bandspec
from libs.geo.chips import chip_raster
from libs.geo.labels import build_label_mask, write_label
from libs.inference.sliding_window import read_tile
from libs.synthetic import make_reflectance_cube, site_lonlat, write_composite
from services.training.dataset import LaunchSiteDataset


def test_dataset_and_inference_preprocessing_are_byte_equal(small_composite, tmp_path):
    path, site = small_composite
    lon, lat = site_lonlat(site)
    sites = [{"name": "s", "lon": lon, "lat": lat, "buffer_km": 0.3}]
    chips = chip_raster(path, tmp_path / "tiles", tile_size=128, overlap=32)
    for c in chips:
        with rasterio.open(c["path"]) as src:
            mask = build_label_mask(src, sites)
            c["label_path"] = str(
                write_label(tmp_path / "tiles" / f"{c['tile_id']}_label.tif", mask, src)
            )
        c["split"] = "train"
    stats = compute_channel_stats([path], sample_px=5000)
    stats_path = tmp_path / "stats.json"
    save_stats(stats, stats_path)

    ds = LaunchSiteDataset(chips, stats_path, split="train", augment=False)
    chip = ds.chips[len(ds) // 2]
    sample = ds[len(ds) // 2]

    with rasterio.open(path) as src:
        names = read_bandspec(src).names
        tile = read_tile(src, chip["row_off"], chip["col_off"], 128, names, ds.stats)

    assert np.array_equal(sample["image"].numpy(), tile)


def test_parity_holds_for_every_chip_with_nan_pixels(tmp_path):
    cube = make_reflectance_cube(320, 320, 160, 160, 30)
    cube[:, 10:14, 10:14] = np.nan  # cloud hole inside the first chip
    path = write_composite(tmp_path / "nan.tif", cube)
    chips = chip_raster(path, tmp_path / "tiles", tile_size=128, overlap=32)
    assert chips
    stats = compute_channel_stats([path], sample_px=5000)
    stats_path = tmp_path / "stats.json"
    save_stats(stats, stats_path)
    for c in chips:
        c["split"] = "train"
        c["label_path"] = str(tmp_path / f"{c['tile_id']}_label.tif")
        with rasterio.open(c["path"]) as src:
            write_label(tmp_path / f"{c['tile_id']}_label.tif", np.zeros((128, 128), "uint8"), src)

    ds = LaunchSiteDataset(chips, stats_path, split="train", augment=False)
    assert len(ds) == len(chips)
    with rasterio.open(path) as src:
        names = read_bandspec(src).names
        for i, chip in enumerate(ds.chips):
            tile = read_tile(src, chip["row_off"], chip["col_off"], 128, names, ds.stats)
            assert np.array_equal(ds[i]["image"].numpy(), tile), chip["tile_id"]
            assert not np.isnan(tile).any()
