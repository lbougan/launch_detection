"""`python -m services.training.evaluate` end to end, with a fake predictor in place of a checkpoint."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from libs.features.stats import compute_channel_stats, save_stats
from libs.inference.predictors import FakePredictor, LocalTorchPredictor
from libs.synthetic import make_synthetic_composite, site_lonlat
from services.training import evaluate


def _setup(tmp_path: Path) -> dict:
    comps = tmp_path / "composites"
    comp, site = make_synthetic_composite(comps / "aoi_x_composite.tif", 256, 256, 24, seed=4)
    lon, lat = site_lonlat(site)
    sites = tmp_path / "sites.json"
    sites.write_text(json.dumps({"sites": [{"name": "s", "lon": lon, "lat": lat}]}))
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"train": [], "val": [], "test": ["s"], "exclusion_km": 20}))
    aoi_config = tmp_path / "aoi_config.json"
    aoi_bbox = [lon - 0.02, lat - 0.02, lon + 0.02, lat + 0.02]
    aoi_config.write_text(json.dumps({"aois": [{"name": "aoi_x", "bbox": aoi_bbox}]}))
    stats = tmp_path / "stats.json"
    save_stats(compute_channel_stats([comp], sample_px=5_000), stats)
    readme = tmp_path / "README.md"
    readme.write_text(f"# Model\n\n{evaluate.START_MARK}\nold\n{evaluate.END_MARK}\n\ntail\n")
    return {
        "comps": comps,
        "sites": sites,
        "splits": splits,
        "aoi_config": aoi_config,
        "stats": stats,
        "readme": readme,
    }


def _argv(p: dict, out: Path) -> list[str]:
    argv = ["evaluate", "--checkpoint", "fake.ckpt", "--aoi-config", str(p["aoi_config"])]
    argv += ["--sites", str(p["sites"]), "--splits", str(p["splits"])]
    return argv + ["--composites-dir", str(p["comps"]), "--out-dir", str(out), "--split", "test"]


def test_main_writes_results_and_readme_block(tmp_path: Path, monkeypatch):
    p = _setup(tmp_path)
    stats, readme = p["stats"], p["readme"]
    aoi_config, sites, splits, comps = p["aoi_config"], p["sites"], p["splits"], p["comps"]
    monkeypatch.setattr(LocalTorchPredictor, "from_checkpoint", lambda *a, **k: FakePredictor(0.9))
    out = tmp_path / "eval"
    argv = ["evaluate", "--checkpoint", "fake.ckpt", "--aoi-config", str(aoi_config)]
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


def test_split_is_selected_by_aoi_not_by_stray_chips(tmp_path: Path, monkeypatch):
    """A training AOI never enters the held-out evaluation, whatever its chips were labelled."""
    p = _setup(tmp_path)
    train_comp, _ = make_synthetic_composite(p["comps"] / "aoi_train_composite.tif", 128, 128, 16)
    cfg = json.loads(p["aoi_config"].read_text())
    cfg["aois"].append({"name": "aoi_train", "bbox": [20.0, 50.0, 20.1, 50.1]})
    p["aoi_config"].write_text(json.dumps(cfg))
    sites = json.loads(p["sites"].read_text())
    sites["sites"].append({"name": "far", "lon": 20.05, "lat": 50.05})
    p["sites"].write_text(json.dumps(sites))
    splits = json.loads(p["splits"].read_text())
    splits["train"] = ["far"]
    p["splits"].write_text(json.dumps(splits))
    seen: list[list[Path]] = []
    real = evaluate.evaluate_composites

    def spy(predictor, comps, *a, **k):
        seen.append(list(comps))
        return real(predictor, comps, *a, **k)

    monkeypatch.setattr(evaluate, "evaluate_composites", spy)
    monkeypatch.setattr(LocalTorchPredictor, "from_checkpoint", lambda *a, **k: FakePredictor(0.9))
    monkeypatch.setattr(sys, "argv", _argv(p, tmp_path / "eval") + ["--stats", str(p["stats"])])
    evaluate.main()
    assert [c.name for c in seen[0]] == ["aoi_x_composite.tif"]
