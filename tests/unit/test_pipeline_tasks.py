"""Pipeline task bodies run on synthetic data (Prefect's `.fn` skips the orchestration layer)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from libs.inference.predictors import FakePredictor
from libs.synthetic import CRS_UTM33, make_synthetic_composite, site_lonlat
from services.pipeline import tasks


class ClosablePredictor(FakePredictor):
    closed = False

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def composite(tmp_path: Path):
    return make_synthetic_composite(tmp_path / "comp.tif", 320, 320, 24, seed=3)


@pytest.fixture
def known_sites(tmp_path: Path, composite) -> Path:
    lon, lat = site_lonlat(composite[1])
    path = tmp_path / "known_sites.json"
    path.write_text(
        json.dumps({"sites": [{"name": "s", "lon": lon, "lat": lat, "buffer_km": 0.2}]})
    )
    return path


def test_build_composite_writes_median_with_bandspec(tmp_path: Path, composite):
    out = tasks.build_composite_task.fn([str(composite[0])] * 3, str(tmp_path / "out" / "c.tif"))
    with rasterio.open(out) as src, rasterio.open(composite[0]) as ref:
        np.testing.assert_allclose(src.read(), ref.read())
        assert src.tags().keys() == ref.tags().keys()


def test_build_composite_rejects_no_scenes(tmp_path: Path):
    with pytest.raises(ValueError, match="No scenes"):
        tasks.build_composite_task.fn([], str(tmp_path / "c.tif"))


def test_stats_tile_split_label_chain(tmp_path: Path, composite, known_sites):
    stats = tasks.compute_stats_task.fn([str(composite[0])], str(tmp_path / "stats.json"))
    assert "n_samples" in json.loads(Path(stats).read_text())

    chips = tasks.tile_composite_task.fn(str(composite[0]), str(tmp_path / "tiles"), 128, 32)
    assert len(chips) == 9

    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"train": [], "val": ["s"], "test": [], "exclusion_km": 20}))
    split_chips = tasks.assign_splits_task.fn(chips, str(known_sites), str(splits))
    assert {c["split"] for c in split_chips} == {"val"}

    labelled = tasks.build_labels_task.fn(split_chips, str(known_sites))
    assert all("label_path" not in c for c in split_chips), "inputs are not mutated"
    positives = 0
    for c in labelled:
        with rasterio.open(c["label_path"]) as src:
            positives += int(src.read(1).sum())
    assert positives > 0


def test_run_inference_local_writes_prob_and_closes(tmp_path: Path, composite, monkeypatch):
    from libs.inference.predictors import LocalTorchPredictor

    predictor = ClosablePredictor(0.25)
    monkeypatch.setattr(LocalTorchPredictor, "from_checkpoint", lambda *a, **k: predictor)
    stats = tasks.compute_stats_task.fn([str(composite[0])], str(tmp_path / "stats.json"))
    out = tasks.run_inference_task.fn(
        str(composite[0]), str(tmp_path / "prob.tif"), stats, "local", checkpoint_path="x.ckpt"
    )
    with rasterio.open(out) as src:
        assert float(np.nanmax(src.read(1))) == pytest.approx(0.25)
    assert predictor.closed


def test_run_inference_uses_checkpoint_stats_and_rejects_a_different_file(
    tmp_path: Path, composite, monkeypatch
):
    from libs.features.stats import compute_channel_stats, save_stats
    from libs.inference.predictors import LocalTorchPredictor

    stats = compute_channel_stats([composite[0]], sample_px=2000)
    predictor = ClosablePredictor(0.25)
    predictor.stats = stats
    monkeypatch.setattr(LocalTorchPredictor, "from_checkpoint", lambda *a, **k: predictor)
    out = tasks.run_inference_task.fn(
        str(composite[0]), str(tmp_path / "prob.tif"), "", "local", checkpoint_path="x.ckpt"
    )
    assert Path(out).exists(), "no stats file needed when the checkpoint carries stats"
    other = tmp_path / "other.json"
    save_stats({**stats, "mean": [m + 1 for m in stats["mean"]]}, other)
    with pytest.raises(ValueError, match="checkpoint"):
        tasks.run_inference_task.fn(
            str(composite[0]), str(tmp_path / "p2.tif"), str(other), "local", checkpoint_path="x"
        )
    assert predictor.closed


def test_run_inference_triton_uses_env_url(tmp_path: Path, composite, monkeypatch):
    from libs.inference.predictors import TritonPredictor

    seen = {}

    def connect(url):
        seen["url"] = url
        return FakePredictor(0.1)

    monkeypatch.setattr(TritonPredictor, "connect", staticmethod(connect))
    monkeypatch.setattr(tasks, "TRITON_URL", "triton:8001")
    stats = tasks.compute_stats_task.fn([str(composite[0])], str(tmp_path / "stats.json"))
    tasks.run_inference_task.fn(str(composite[0]), str(tmp_path / "prob.tif"), stats, "triton")
    assert seen["url"] == "triton:8001"


@pytest.mark.parametrize(
    ("mode", "match"),
    [("triton", "Triton URL not set"), ("local", "checkpoint_path"), ("onnx", "Unknown serving")],
)
def test_run_inference_rejects_bad_config(tmp_path: Path, monkeypatch, mode, match):
    monkeypatch.setattr(tasks, "TRITON_URL", "")
    with pytest.raises(ValueError, match=match):
        tasks.run_inference_task.fn("r.tif", str(tmp_path / "p.tif"), "s.json", mode)


def _write_prob(path: Path, value: float) -> Path:
    prob = np.zeros((200, 200), dtype=np.float32)
    prob[80:120, 80:120] = value
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": 200,
        "height": 200,
        "count": 1,
        "crs": CRS_UTM33,
        "transform": from_origin(500_000.0, 4_000_000.0, 10.0, 10.0),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(prob, 1)
    return path


def test_postprocess_returns_one_feature_per_site(tmp_path: Path):
    fc = tasks.postprocess_task.fn(str(_write_prob(tmp_path / "p.tif", 0.9)))
    assert len(fc["features"]) == 1
    assert fc["features"][0]["properties"]["prob"] == pytest.approx(0.9)


def test_postprocess_feature_coordinates_are_lonlat(tmp_path: Path):
    from pyproj import Transformer

    fc = tasks.postprocess_task.fn(str(_write_prob(tmp_path / "p.tif", 0.9)))
    ring = fc["features"][0]["geometry"]["coordinates"][0]
    assert all(-180 <= x <= 180 and -90 <= y <= 90 for x, y in ring)
    # blob centre: pixel (100, 100) of a 10 m grid at (500000, 4000000)
    lon, lat = Transformer.from_crs(CRS_UTM33, "EPSG:4326", always_xy=True).transform(
        501_000.0, 3_999_000.0
    )
    xs, ys = [x for x, _ in ring], [y for _, y in ring]
    assert min(xs) < lon < max(xs) and min(ys) < lat < max(ys)


def test_postprocess_empty_when_nothing_crosses_threshold(tmp_path: Path):
    fc = tasks.postprocess_task.fn(str(_write_prob(tmp_path / "p.tif", 0.2)))
    assert fc == {"type": "FeatureCollection", "features": []}


def test_train_model_builds_fit_config(monkeypatch):
    import services.training.fit as fit_mod

    seen = {}

    def fake_fit(cfg):
        seen["cfg"] = cfg
        return "best.ckpt"

    monkeypatch.setattr(fit_mod, "fit", fake_fit)
    best = tasks.train_model_task.fn("m.json", "s.json", gpus=1, epochs=3)
    assert best == "best.ckpt"
    cfg = seen["cfg"]
    assert (cfg.manifest_path, cfg.stats_path, cfg.epochs) == ("m.json", "s.json", 3)
    assert cfg.precision == "16-mixed"
