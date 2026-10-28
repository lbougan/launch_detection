"""Composite -> chips -> labels -> stats -> 2 epochs on CPU -> inference -> postprocess -> recall@5.

Runs in well under two minutes on 4 cores. Marked integration (collected by default) and
measured by CI as the proof that the stages fit together.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
import rasterio

from libs.features.stats import compute_channel_stats, save_stats
from libs.geo.chips import chip_raster
from libs.geo.labels import build_label_mask, write_label
from libs.geo.splits import assign_splits
from libs.inference.predictors import LocalTorchPredictor
from services.training.evaluate import evaluate_composites
from services.training.fit import FitConfig, fit
from tests.fixtures.synthetic import make_synthetic_composite, site_lonlat

pytestmark = pytest.mark.integration

SHIFT_EAST_M = 200_000.0


def _site_dict(name: str, site) -> dict:
    lon, lat = site_lonlat(site)
    return {"name": name, "lon": lon, "lat": lat, "buffer_km": 0.3}


def test_planted_site_is_recovered_end_to_end(tmp_path: Path):
    t0 = time.perf_counter()
    train_comp, train_site = make_synthetic_composite(tmp_path / "train.tif", 768, 768, 40, seed=1)
    val_comp, val_site = make_synthetic_composite(tmp_path / "val.tif", 512, 512, 40, seed=2)
    # Two composites far apart in the world: shift the validation one 200 km east.
    with rasterio.open(val_comp, "r+") as dst:
        t = dst.transform
        dst.transform = rasterio.Affine(t.a, t.b, t.c + SHIFT_EAST_M, t.d, t.e, t.f)
    val_site = type(val_site)(
        val_site.row, val_site.col, val_site.size_px, val_site.x + SHIFT_EAST_M, val_site.y
    )
    sites = [_site_dict("train_site", train_site), _site_dict("val_site", val_site)]
    splits = {"train": ["train_site"], "val": ["val_site"], "test": []}

    chips = chip_raster(train_comp, tmp_path / "tiles_train", tile_size=128, overlap=32)
    chips += chip_raster(val_comp, tmp_path / "tiles_val", tile_size=128, overlap=32)
    chips = assign_splits(chips, sites, splits, exclusion_km=20.0)
    labelled = []
    for c in chips:
        with rasterio.open(c["path"]) as src:
            label_path = Path(c["path"]).with_name(c["tile_id"] + "_label.tif")
            written = write_label(label_path, build_label_mask(src, sites), src)
        labelled.append({**c, "label_path": str(written)})
    assert {c["split"] for c in labelled} >= {"train", "val"}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(labelled))
    stats = tmp_path / "stats.json"
    save_stats(compute_channel_stats([train_comp], sample_px=20_000), stats)

    best = fit(
        FitConfig(
            manifest_path=str(manifest),
            stats_path=str(stats),
            encoder="resnet18",
            encoder_weights=None,
            epochs=2,
            batch_size=8,
            num_workers=0,
            checkpoint_dir=str(tmp_path / "ckpt"),
            mlflow_tracking_uri=None,
            val_composites=(),
        )
    )
    assert Path(best).exists()

    predictor = LocalTorchPredictor.from_checkpoint(best)
    result = evaluate_composites(
        predictor, [val_comp], stats, [sites[1]], tmp_path / "eval", "val", best, threshold=0.5
    )
    with rasterio.open(tmp_path / "eval" / "val_prob.tif") as src:
        prob_max = float(np.nanmax(src.read(1)))
    assert result.recall_at_5 == 1.0, f"planted site not in top 5 (prob max {prob_max}): {result}"
    assert result.fp_per_1000km2 < 100, result
    assert time.perf_counter() - t0 < 120, "integration test must stay under two minutes"
