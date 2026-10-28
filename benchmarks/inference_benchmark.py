"""Benchmark harness: compare baseline PyTorch vs. Triton batched vs. memory-optimized inference.

Usage:
    python -m benchmarks.inference_benchmark \
        --raster path/to/composite.tif \
        --checkpoint checkpoints/best.ckpt \
        --triton-url localhost:8001 \
        --modes baseline,triton,triton_memopt \
        --batch-sizes 1,8,16,32 \
        --overlaps 0,32,64

Modes: baseline (local torch), triton, triton_memopt (triton + memmap accumulator), plus
fake / fake_memmap (constant predictor, no model or server) for smoke tests.

Results are written to benchmarks/results/ as JSON (not written, exit code 1, if nothing ran).

Caveats: ``peak_rss_mb`` is process-wide (ru_maxrss), so later modes include the peaks of earlier
ones. ``peak_tracemalloc_mb`` does not count memmap pages (they live in the OS page cache).
"""

from __future__ import annotations

import argparse
import json
import logging
import resource
import tempfile
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import rasterio

from libs.features.stats import compute_channel_stats, load_stats
from libs.inference.sliding_window import iter_windows, run_sliding_window

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    mode: str
    raster_size: tuple[int, int]
    tile_size: int
    overlap: int
    batch_size: int
    n_tiles: int
    wall_time_s: float
    tiles_per_sec: float
    peak_rss_mb: float
    peak_tracemalloc_mb: float
    batch_latencies_ms: list[float] = field(default_factory=list)

    @property
    def p50_ms(self) -> float:
        return float(np.percentile(self.batch_latencies_ms, 50)) if self.batch_latencies_ms else 0.0

    @property
    def p95_ms(self) -> float:
        return float(np.percentile(self.batch_latencies_ms, 95)) if self.batch_latencies_ms else 0.0

    @property
    def p99_ms(self) -> float:
        return float(np.percentile(self.batch_latencies_ms, 99)) if self.batch_latencies_ms else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["p50_ms"] = self.p50_ms
        d["p95_ms"] = self.p95_ms
        d["p99_ms"] = self.p99_ms
        return d


def create_synthetic_raster(size: int, tmpdir: str) -> Path:
    """Synthetic 6-band composite (planted site, BANDSPEC tag) written via the test fixtures."""
    from tests.fixtures.synthetic import make_reflectance_cube, write_composite

    cube = make_reflectance_cube(size, size, size // 2, size // 2, max(8, size // 25))
    return write_composite(Path(tmpdir) / f"synthetic_{size}x{size}.tif", cube)


def _count_tiles(h: int, w: int, tile_size: int, overlap: int) -> int:
    return sum(1 for _ in iter_windows(h, w, tile_size, overlap))


def _measure_rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_maxrss / 1024  # Linux reports KB


class _TimedPredictor:
    """Wraps a predictor and records per-batch latency in milliseconds."""

    def __init__(self, inner):
        self.inner = inner
        self.latencies_ms: list[float] = []

    def predict(self, batch: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        out = self.inner.predict(batch)
        self.latencies_ms.append((time.perf_counter() - t0) * 1000)
        return out


def benchmark_mode(
    name: str,
    predictor,
    memmap: bool,
    raster_path: Path,
    stats: dict,
    output_dir: Path,
    tile_size: int,
    overlap: int,
    batch_size: int,
) -> BenchmarkResult:
    """Benchmark one (predictor, memmap) pair over the shared sliding-window implementation."""
    raster_path, output_dir = Path(raster_path), Path(output_dir)
    h, w = _raster_dims(raster_path)
    n_tiles = _count_tiles(h, w, tile_size, overlap)
    timed = _TimedPredictor(predictor)

    tracemalloc.start()
    t0 = time.perf_counter()
    run_sliding_window(
        raster_path=raster_path,
        output_path=output_dir / f"{name}_prob.tif",
        predictor=timed,
        stats=stats,
        tile=tile_size,
        overlap=overlap,
        batch_size=batch_size,
        memmap=memmap,
        tmpdir=output_dir,
    )
    wall = time.perf_counter() - t0
    _, peak_tm = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return BenchmarkResult(
        mode=name,
        raster_size=(h, w),
        tile_size=tile_size,
        overlap=overlap,
        batch_size=batch_size,
        n_tiles=n_tiles,
        wall_time_s=wall,
        tiles_per_sec=n_tiles / wall if wall > 0 else 0,
        peak_rss_mb=_measure_rss_mb(),
        peak_tracemalloc_mb=peak_tm / (1024 * 1024),
        batch_latencies_ms=timed.latencies_ms,
    )


def _build_mode(mode: str, args) -> tuple[object, bool] | None:
    """Return (predictor, memmap) for a mode name, or None if unknown."""
    if mode in ("fake", "fake_memmap"):
        from libs.inference.predictors import FakePredictor

        return FakePredictor(), mode == "fake_memmap"
    if mode == "baseline":
        from libs.inference.predictors import LocalTorchPredictor

        return LocalTorchPredictor.from_checkpoint(args.checkpoint, device=args.device), False
    if mode in ("triton", "triton_memopt"):
        from libs.inference.predictors import TritonPredictor

        return TritonPredictor.connect(args.triton_url), mode == "triton_memopt"
    return None


def _raster_dims(path: Path) -> tuple[int, int]:
    with rasterio.open(path) as src:
        return src.height, src.width


def _print_table(results: list[BenchmarkResult]) -> None:
    header = (
        f"{'Mode':<22} {'Size':>12} {'Tiles':>6} {'Batch':>5} "
        f"{'Wall(s)':>8} {'Tile/s':>8} {'RSS(MB)':>8} "
        f"{'p50(ms)':>8} {'p95(ms)':>8} {'p99(ms)':>8}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for r in results:
        size_str = f"{r.raster_size[0]}x{r.raster_size[1]}"
        print(
            f"{r.mode:<22} {size_str:>12} {r.n_tiles:>6} {r.batch_size:>5} "
            f"{r.wall_time_s:>8.2f} {r.tiles_per_sec:>8.1f} {r.peak_rss_mb:>8.0f} "
            f"{r.p50_ms:>8.1f} {r.p95_ms:>8.1f} {r.p99_ms:>8.1f}"
        )
    print("=" * len(header) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference benchmark harness")
    parser.add_argument("--raster", help="Path to raster (omit to use synthetic)")
    parser.add_argument("--stats", help="stats.json (required with --raster)")
    parser.add_argument("--checkpoint", default="checkpoints/best.ckpt")
    parser.add_argument("--triton-url", default="localhost:8001")
    parser.add_argument(
        "--modes",
        default="baseline,triton,triton_memopt",
        help="Comma-separated list of modes",
    )
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--batch-sizes", default="1,8,16,32", help="Comma-separated batch sizes")
    parser.add_argument("--overlaps", default="0,32,64", help="Comma-separated overlap values")
    parser.add_argument(
        "--raster-sizes",
        default="1024,4096,11000",
        help="Comma-separated synthetic raster edge lengths (used when --raster is omitted)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="benchmarks/results")

    args = parser.parse_args()
    if args.raster and not args.stats:
        parser.error("--stats is required with --raster")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    modes = [m.strip() for m in args.modes.split(",")]
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    overlaps = [int(o) for o in args.overlaps.split(",")]
    raster_sizes = [int(s) for s in args.raster_sizes.split(",")]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[BenchmarkResult] = []

    with tempfile.TemporaryDirectory(prefix="bench_") as tmpdir:
        if args.raster:
            raster_paths = {0: Path(args.raster)}
        else:
            raster_paths = {}
            for size in raster_sizes:
                logger.info("Creating synthetic raster %dx%d ...", size, size)
                raster_paths[size] = create_synthetic_raster(size, tmpdir)

        work_dir = Path(tmpdir) / "outputs"
        work_dir.mkdir()

        built: dict[str, tuple[object, bool]] = {}
        for mode in modes:
            try:
                pair = _build_mode(mode, args)
            except Exception:
                logger.exception("Could not set up mode %s", mode)
                continue
            if pair is None:
                logger.warning("Unknown mode: %s", mode)
                continue
            built[mode] = pair

        for size_key, raster_path in raster_paths.items():
            stats = (
                load_stats(Path(args.stats))
                if args.stats
                else compute_channel_stats([raster_path], sample_px=20000)
            )
            for ov in overlaps:
                for bs in batch_sizes:
                    for mode, (predictor, memmap) in built.items():
                        label = f"{mode}|size={size_key}|ov={ov}|bs={bs}"
                        logger.info("Running: %s", label)
                        try:
                            result = benchmark_mode(
                                mode,
                                predictor,
                                memmap,
                                raster_path,
                                stats,
                                work_dir,
                                args.tile_size,
                                ov,
                                bs,
                            )
                            all_results.append(result)
                            logger.info(
                                "  -> %.2fs, %.1f tiles/s, %.0f MB RSS",
                                result.wall_time_s,
                                result.tiles_per_sec,
                                result.peak_rss_mb,
                            )
                        except Exception:
                            logger.exception("  -> FAILED: %s", label)

    if not built or not all_results:
        logger.error("No benchmark results produced; not writing a results file")
        raise SystemExit(1)

    _print_table(all_results)

    results_file = output_dir / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump([r.to_dict() for r in all_results], f, indent=2)
    logger.info("Results saved → %s", results_file)


if __name__ == "__main__":
    main()
