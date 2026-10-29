import pytest

from services.training.fit import FitConfig


def test_fit_config_defaults_are_cpu_friendly():
    cfg = FitConfig(manifest_path="m.json", stats_path="s.json")
    assert cfg.gpus == 0 and cfg.precision == 32 and cfg.num_workers >= 0
    assert cfg.monitor == "val/site_recall@10"


def test_checkpoint_filename_carries_the_monitored_metric():
    from pytorch_lightning.callbacks import ModelCheckpoint

    from services.training.fit import checkpoint_filename

    cb = ModelCheckpoint(
        monitor="val/site_recall@10",
        filename=checkpoint_filename("val/site_recall@10"),
        auto_insert_metric_name=False,
    )
    name = cb.format_checkpoint_name({"epoch": 3, "val/site_recall@10": 0.457})
    assert name.endswith("launchsite-03-0.457.ckpt")


def test_recall_monitor_falls_back_to_val_loss_without_val_composites():
    from services.training.fit import resolve_monitor

    cfg = FitConfig(manifest_path="m.json", stats_path="s.json")
    assert resolve_monitor(cfg) == ("val/loss", "min")
    with_comps = FitConfig(manifest_path="m", stats_path="s", val_composites=("c.tif",))
    assert resolve_monitor(with_comps) == ("val/site_recall@10", "max")
    loss = FitConfig(manifest_path="m", stats_path="s", monitor="val/loss")
    assert resolve_monitor(loss) == ("val/loss", "min")


def test_fit_rejects_missing_val_composites_before_training(tmp_path, monkeypatch):
    import pytorch_lightning as pl

    from services.training.fit import fit

    def boom(*a, **k):
        raise AssertionError("Trainer must not be built")

    monkeypatch.setattr(pl, "Trainer", boom)
    present = tmp_path / "ok.tif"
    present.write_bytes(b"")
    cfg = FitConfig(
        manifest_path=str(tmp_path / "m.json"),
        stats_path="s.json",
        val_composites=(str(present), str(tmp_path / "nope.tif")),
    )
    with pytest.raises(FileNotFoundError, match="nope.tif") as exc:
        fit(cfg)
    assert "ok.tif" not in str(exc.value)
