from pathlib import Path

from libs.config import default_stats_path, settings


def test_default_stats_path_follows_data_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    assert default_stats_path() == tmp_path / "manifests" / "channel_stats.json"
    monkeypatch.setattr(settings, "data_dir", Path("elsewhere"))
    assert default_stats_path() == Path("elsewhere/manifests/channel_stats.json")
