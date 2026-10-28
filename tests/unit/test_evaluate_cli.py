"""`python -m services.training.evaluate` end to end, with a fake predictor in place of a checkpoint."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from libs.features.stats import compute_channel_stats, save_stats
from libs.inference.predictors import FakePredictor, LocalTorchPredictor
from services.training import evaluate
from tests.fixtures.synthetic import make_synthetic_composite, site_lonlat


def test_main_writes_results_and_readme_block(tmp_path: Path, monkeypatch):
    comps = tmp_path / "composites"
    comp, site = make_synthetic_composite(comps / "aoi_x_composite.tif", 256, 256, 24, seed=4)
    lon, lat = site_lonlat(site)
    sites = tmp_path / "sites.json"
    sites.write_text(json.dumps({"sites": [{"name": "s", "lon": lon, "lat": lat}]}))
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"train": [], "val": [], "test": ["s"], "exclusion_km": 20}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([{"tile_id": "t", "aoi": "aoi_x", "split": "test"}]))
    stats = tmp_path / "stats.json"
    save_stats(compute_channel_stats([comp], sample_px=5_000), stats)
    readme = tmp_path / "README.md"
    readme.write_text(f"# Model\n\n{evaluate.START_MARK}\nold\n{evaluate.END_MARK}\n\ntail\n")

    monkeypatch.setattr(LocalTorchPredictor, "from_checkpoint", lambda *a, **k: FakePredictor(0.9))
    out = tmp_path / "eval"
    argv = ["evaluate", "--checkpoint", "fake.ckpt", "--manifest", str(manifest)]
    argv += ["--stats", str(stats), "--sites", str(sites), "--splits", str(splits)]
    argv += ["--composites-dir", str(comps), "--out-dir", str(out), "--split", "test"]
    argv += ["--update-readme", str(readme)]
    monkeypatch.setattr(sys, "argv", argv)

    evaluate.main()

    result = json.loads((out / "test.json").read_text())
    assert result["split"] == "test" and result["n_sites"] == 1
    assert result["checkpoint"] == "fake.ckpt"
    text = readme.read_text()
    assert "old" not in text and text.endswith(f"{evaluate.END_MARK}\n\ntail\n")
    block = text[text.index(evaluate.START_MARK) : text.index(evaluate.END_MARK)]
    assert "fake.ckpt" in block and "test" in block
