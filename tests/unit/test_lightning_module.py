"""Site-level validation in the Lightning module: fail fast, rank-0 only."""

from types import SimpleNamespace

import pytest

from services.training.lightning_module import LaunchSiteSegModule


def _module(monkeypatch, is_global_zero=True):
    m = LaunchSiteSegModule(encoder_name="resnet18", encoder_weights=None, val_composites=["c.tif"])
    trainer = SimpleNamespace(sanity_checking=False, is_global_zero=is_global_zero)
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


def test_site_metrics_logged_on_rank_zero_without_sync(monkeypatch):
    m, logged = _module(monkeypatch)
    monkeypatch.setattr(m, "_site_metrics", lambda: (0.5, 3.0))
    m._log_site_metrics()
    assert [(n, v) for n, v, _ in logged] == [
        ("val/site_recall@10", 0.5),
        ("val/fp_per_1000km2", 3.0),
    ]
    assert all(kw["rank_zero_only"] and kw["sync_dist"] is False for _, _, kw in logged)


def test_other_ranks_neither_compute_nor_log(monkeypatch):
    m, logged = _module(monkeypatch, is_global_zero=False)
    monkeypatch.setattr(m, "_site_metrics", lambda: pytest.fail("rank > 0 must not evaluate"))
    m._log_site_metrics()
    assert logged == []
