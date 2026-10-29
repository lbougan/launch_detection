import json

import onnx
import pytest
import torch

from services.serving.export_model import export_to_onnx
from services.training.lightning_module import LaunchSiteSegModule


def test_export_embeds_stats_and_bandspec(tmp_path):
    module = LaunchSiteSegModule(
        model_type="unet", encoder_name="resnet18", encoder_weights=None, in_channels=10, lr=1e-3
    )
    ckpt = tmp_path / "m.ckpt"
    torch.save({"state_dict": module.state_dict(), "hyper_parameters": dict(module.hparams)}, ckpt)
    stats = tmp_path / "stats.json"
    stats.write_text('{"channels": ["a"], "p2": [0], "p98": [1], "mean": [0.5], "std": [0.5]}')
    out = export_to_onnx(
        ckpt,
        tmp_path / "model.onnx",
        stats_path=stats,
        validate=False,
        in_channels=10,
        tile_size=64,
    )
    meta = {p.key: p.value for p in onnx.load(str(out)).metadata_props}
    assert "stats_json" in meta and '"channels"' in meta["stats_json"]
    assert "bandspec_json" in meta and "git_commit" in meta
    assert meta["in_channels"] == "10" and meta["tile_size"] == "64"
    assert (tmp_path / "stats.json").exists()


STATS = {"channels": ["a"], "p2": [0], "p98": [1], "mean": [0.5], "std": [0.5]}


def _checkpoint(tmp_path, stats):
    module = LaunchSiteSegModule(
        model_type="unet",
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=10,
        lr=1e-3,
        stats=stats,
    )
    ckpt = tmp_path / "m.ckpt"
    torch.save({"state_dict": module.state_dict(), "hyper_parameters": dict(module.hparams)}, ckpt)
    return ckpt


def test_export_ships_the_checkpoints_own_stats(tmp_path):
    ckpt = _checkpoint(tmp_path, STATS)
    out = export_to_onnx(
        ckpt, tmp_path / "model.onnx", validate=False, in_channels=10, tile_size=64
    )
    meta = {p.key: p.value for p in onnx.load(str(out)).metadata_props}
    assert json.loads(meta["stats_json"]) == STATS
    assert json.loads((tmp_path / "stats.json").read_text()) == STATS


def test_export_rejects_a_stats_file_that_differs_from_the_checkpoint(tmp_path):
    """Re-running ingest rewrites channel_stats.json; Triton must not normalise with it."""
    ckpt = _checkpoint(tmp_path, STATS)
    other = tmp_path / "other.json"
    other.write_text(json.dumps({**STATS, "std": [1.0]}))
    with pytest.raises(ValueError, match="checkpoint"):
        export_to_onnx(
            ckpt, tmp_path / "model.onnx", stats_path=other, validate=False, tile_size=64
        )
    assert not (tmp_path / "model.onnx").exists(), "checked before the export runs"
