from services.training.fit import FitConfig


def test_fit_config_defaults_are_cpu_friendly():
    cfg = FitConfig(manifest_path="m.json", stats_path="s.json")
    assert cfg.gpus == 0 and cfg.precision == 32 and cfg.num_workers >= 0
    assert cfg.monitor == "val/site_recall@10"
