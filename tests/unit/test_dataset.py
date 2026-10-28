import numpy as np
import rasterio
import torch

from libs.features.stats import compute_channel_stats, save_stats
from libs.geo.chips import chip_raster
from libs.geo.labels import build_label_mask, write_label
from services.training.dataset import LaunchSiteDataset
from tests.fixtures.synthetic import site_lonlat


def test_dataset_returns_normalised_image_mask_and_weight(small_composite, tmp_path):
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
    stats_path = tmp_path / "stats.json"
    save_stats(compute_channel_stats([path], sample_px=5000), stats_path)
    ds = LaunchSiteDataset(chips, stats_path, split="train", augment=False)
    sample = ds[0]
    assert sample["image"].shape == (10, 128, 128)
    assert sample["mask"].shape == (1, 128, 128)
    assert sample["weight"].shape == (1, 128, 128)
    assert sample["image"].dtype == torch.float32
    assert not torch.isnan(sample["image"]).any()
    assert abs(float(sample["image"].mean())) < 3.0  # standardised
    assert np.isin(sample["mask"].numpy(), [0.0, 1.0]).all()
