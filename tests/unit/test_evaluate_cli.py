"""`python -m services.training.evaluate` end to end, with a fake predictor in place of a checkpoint."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from libs.features.stats import compute_channel_stats, save_stats
from libs.inference.predictors import FakePredictor, LocalTorchPredictor
from services.training import evaluate
from tests.fixtures.synthetic import make_synthetic_composite, site_lonlat


def _setup(tmp_path: Path) -> dict:
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
    return {
        "comps": comps,
        "sites": sites,
        "splits": splits,
        "manifest": manifest,
        "stats": stats,
        "readme": readme,
    }


def _argv(p: dict, out: Path) -> list[str]:
    argv = ["evaluate", "--checkpoint", "fake.ckpt", "--manifest", str(p["manifest"])]
    argv += ["--sites", str(p["sites"]), "--splits", str(p["splits"])]
    return argv + ["--composites-dir", str(p["comps"]), "--out-dir", str(out), "--split", "test"]


def test_main_writes_results_and_readme_block(tmp_path: Path, monkeypatch):
    p = _setup(tmp_path)
    stats, readme = p["stats"], p["readme"]
    manifest, sites, splits, comps = p["manifest"], p["sites"], p["splits"], p["comps"]
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


def test_main_uses_checkpoint_stats_device_and_rejects_a_different_stats_file(
    tmp_path: Path, monkeypatch
):
    p = _setup(tmp_path)
    seen = {}
    predictor = FakePredictor(0.9)
    predictor.stats = json.loads(p["stats"].read_text())

    def from_checkpoint(path, device="cpu"):
        seen["device"] = device
        return predictor

    monkeypatch.setattr(LocalTorchPredictor, "from_checkpoint", from_checkpoint)
    out = tmp_path / "eval"
    monkeypatch.setattr(sys, "argv", _argv(p, out) + ["--device", "cuda:1"])
    evaluate.main()
    assert seen["device"] == "cuda:1"
    assert (out / "test.json").exists(), "no --stats needed: the checkpoint carries them"

    other = tmp_path / "other.json"
    other.write_text(json.dumps({**predictor.stats, "std": [1.0] * len(predictor.stats["std"])}))
    monkeypatch.setattr(sys, "argv", _argv(p, out) + ["--stats", str(other)])
    with pytest.raises(ValueError, match="checkpoint"):
        evaluate.main()
