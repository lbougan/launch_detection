import json
import re
from pathlib import Path

import rasterio

from libs.features.stats import compute_channel_stats, save_stats
from libs.geo.chips import chip_raster
from libs.geo.labels import build_label_mask, write_label
from libs.synthetic import site_lonlat
from services.training.fit import FitConfig, fit_with_trainer


def test_fit_runs_one_epoch_on_cpu_and_logs_site_recall(small_composite, tmp_path):
    path, site = small_composite
    lon, lat = site_lonlat(site)
    sites = [{"name": "s", "lon": lon, "lat": lat, "buffer_km": 0.3}]
    chips = chip_raster(path, tmp_path / "tiles", tile_size=128, overlap=32)
    for i, c in enumerate(chips):
        with rasterio.open(c["path"]) as src:
            mask = build_label_mask(src, sites)
            label = tmp_path / "tiles" / f"{c['tile_id']}_label.tif"
            c["label_path"] = str(write_label(label, mask, src))
        c["split"] = "val" if i % 4 == 0 else "train"
    manifest = tmp_path / "chips.json"
    manifest.write_text(json.dumps(chips))
    stats = tmp_path / "stats.json"
    save_stats(compute_channel_stats([path], sample_px=5000), stats)
    sites_path = tmp_path / "sites.json"
    sites_path.write_text(json.dumps({"sites": sites}))
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"train": [], "val": ["s"], "test": []}))

    cfg = FitConfig(
        manifest_path=str(manifest),
        stats_path=str(stats),
        known_sites_path=str(sites_path),
        splits_path=str(splits),
        encoder="resnet18",
        encoder_weights=None,
        epochs=1,
        batch_size=2,
        num_workers=0,
        checkpoint_dir=str(tmp_path / "ckpt"),
        limit_train_batches=2,
        limit_val_batches=1,
        val_composites=(str(path),),
    )
    best, trainer = fit_with_trainer(cfg)
    assert best and Path(best).exists()
    assert re.search(r"launchsite-00-\d\.\d{3}\.ckpt$", best)
    assert "val/site_recall@10" in trainer.callback_metrics
    import torch

    hparams = torch.load(best, map_location="cpu", weights_only=False)["hyper_parameters"]
    assert hparams["stats"] == json.loads(stats.read_text()), "stats travel with the checkpoint"
    assert hparams["stats_path"] == str(stats)

    from libs.inference.predictors import LocalTorchPredictor

    assert LocalTorchPredictor.from_checkpoint(best).stats == hparams["stats"]
