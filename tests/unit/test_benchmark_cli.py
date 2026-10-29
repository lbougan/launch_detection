import json
import sys

import pytest

from benchmarks.inference_benchmark import main


def test_benchmark_cli_runs_fake_modes(tmp_path, monkeypatch):
    out = tmp_path / "out"
    argv = [
        "inference_benchmark",
        "--modes",
        "fake,fake_memmap",
        "--raster-sizes",
        "256",
        "--batch-sizes",
        "4",
        "--overlaps",
        "32",
        "--output-dir",
        str(out),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    main()
    results = json.loads((out / "benchmark_results.json").read_text())
    assert len(results) == 2
    assert all(r["n_tiles"] > 0 for r in results)


def test_benchmark_cli_exits_nonzero_without_results(tmp_path, monkeypatch):
    out = tmp_path / "out"
    argv = [
        "inference_benchmark",
        "--modes",
        "nonsense",
        "--raster-sizes",
        "256",
        "--output-dir",
        str(out),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert not (out / "benchmark_results.json").exists()
