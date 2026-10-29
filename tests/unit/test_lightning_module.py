"""Site-level validation in the Lightning module: fail fast, computed on rank 0, broadcast."""

from types import SimpleNamespace

import pytest

from services.training.lightning_module import LaunchSiteSegModule


class FakeStrategy:
    """Single-process stand-in: broadcast returns what rank 0 holds (`from_rank0` if set)."""

    def __init__(self, from_rank0=None):
        self.from_rank0 = from_rank0
        self.calls = []

    def broadcast(self, obj, src=0):
        self.calls.append((obj, src))
        return obj if self.from_rank0 is None else self.from_rank0


def _module(monkeypatch, is_global_zero=True, from_rank0=None):
    m = LaunchSiteSegModule(encoder_name="resnet18", encoder_weights=None, val_composites=["c.tif"])
    trainer = SimpleNamespace(
        sanity_checking=False,
        is_global_zero=is_global_zero,
        strategy=FakeStrategy(from_rank0),
    )
    monkeypatch.setattr(LaunchSiteSegModule, "trainer", property(lambda self: trainer))
    logged = []
    monkeypatch.setattr(m, "log", lambda name, value, **kw: logged.append((name, value, kw)))
    return m, logged


def test_site_metric_failure_propagates(monkeypatch):
    m, _ = _module(monkeypatch)

    def fail():
        raise RuntimeError("inference broke")

    monkeypatch.setattr(m, "_site_metrics", fail)
    with pytest.raises(RuntimeError, match="inference broke"):
        m._log_site_metrics()


def test_rank_zero_computes_broadcasts_and_logs_without_sync(monkeypatch):
    m, logged = _module(monkeypatch)
    monkeypatch.setattr(m, "_site_metrics", lambda: (0.5, 3.0))
    m._log_site_metrics()
    assert m.trainer.strategy.calls == [((0.5, 3.0), 0)]
    assert [(n, v) for n, v, _ in logged] == [
        ("val/site_recall@10", 0.5),
        ("val/fp_per_1000km2", 3.0),
    ]
    assert all(kw["sync_dist"] is False and not kw.get("rank_zero_only") for _, _, kw in logged)


def test_other_ranks_log_the_broadcast_value_without_computing(monkeypatch):
    m, logged = _module(monkeypatch, is_global_zero=False, from_rank0=(0.7, 1.0))
    monkeypatch.setattr(m, "_site_metrics", lambda: pytest.fail("rank > 0 must not evaluate"))
    m._log_site_metrics()
    assert [(n, v) for n, v, _ in logged] == [
        ("val/site_recall@10", 0.7),
        ("val/fp_per_1000km2", 1.0),
    ]


def test_default_stats_path_follows_config():
    from libs.config import default_stats_path

    m = LaunchSiteSegModule(encoder_name="resnet18", encoder_weights=None)
    assert m.hparams.stats_path == str(default_stats_path())
