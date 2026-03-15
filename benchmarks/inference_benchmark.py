"""Benchmark harness: compare baseline PyTorch vs. Triton batched vs. memory-optimized inference.

Usage:
    python -m benchmarks.inference_benchmark \
        --raster path/to/composite.tif \
        --checkpoint checkpoints/best.ckpt \
        --triton-url localhost:8001 \
        --modes baseline,triton,triton_memopt \
        --batch-sizes 1,8,16,32 \
        --overlaps 0,32,64

Results are written to benchmarks/results/ as JSON.
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
from typing import Callable

import numpy as np
import rasterio
from rasterio.transform import from_bounds

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


def _gpu_utilization() -> float | None:
    """Sample current GPU utilization percentage, or None if unavailable."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return float(util.gpu)
    except Exception:
        return None


def create_synthetic_raster(
    height: int,
    width: int,
    n_bands: int = 6,
    tmpdir: str | None = None,
) -> Path:
    """Create a synthetic multi-band GeoTIFF for benchmarking."""
    rng = np.random.default_rng(42)
    data = rng.integers(0, 10000, size=(n_bands, height, width), dtype=np.uint16).astype(
        np.float32
    )

    path = Path(tmpdir or tempfile.mkdtemp()) / f"synthetic_{height}x{width}.tif"
    transform = from_bounds(0, 0, width * 10, height * 10, width, height)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": width,
        "height": height,
        "count": n_bands,
        "crs": "EPSG:32633",
        "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)

    return path


def _count_tiles(h: int, w: int, tile_size: int, overlap: int) -> int:
    step = tile_size - overlap
    rows = max(0, (h - tile_size) // step + 1)
    cols = max(0, (w - tile_size) // step + 1)
    return rows * cols


def _measure_rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_maxrss / 1024  # Linux reports KB


def benchmark_baseline(
    raster_path: Path,
    checkpoint_path: str,
    output_dir: Path,
    tile_size: int,
    overlap: int,
    batch_size: int,
    device: str,
) -> BenchmarkResult:
    """Benchmark the original sliding-window inference."""
    from services.training.inference import load_model, sliding_window_inference

    raster_path, output_dir = Path(raster_path), Path(output_dir)
    h, w = _raster_dims(raster_path)
    n_tiles = _count_tiles(h, w, tile_size, overlap)
    out = output_dir / "baseline_prob.tif"

    model = load_model(checkpoint_path, device)

    tracemalloc.start()
    t0 = time.perf_counter()

    sliding_window_inference(
        model=model,
        raster_path=raster_path,
        output_path=out,
        tile_size=tile_size,
        overlap=overlap,
        batch_size=batch_size,
        device=device,
    )

    wall = time.perf_counter() - t0
    _, peak_tm = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return BenchmarkResult(
        mode=f"baseline_{device}",
        raster_size=(h, w),
        tile_size=tile_size,
        overlap=overlap,
        batch_size=batch_size,
        n_tiles=n_tiles,
        wall_time_s=wall,
        tiles_per_sec=n_tiles / wall if wall > 0 else 0,
        peak_rss_mb=_measure_rss_mb(),
        peak_tracemalloc_mb=peak_tm / (1024 * 1024),
    )


def benchmark_triton(
    raster_path: Path,
    triton_url: str,
    output_dir: Path,
    tile_size: int,
    overlap: int,
    batch_size: int,
) -> BenchmarkResult:
    """Benchmark Triton-backed sliding-window inference."""
    from services.serving.triton_client import (
        TritonSegClient,
        _generate_tile_batches,
        _tile_windows,
    )

    raster_path, output_dir = Path(raster_path), Path(output_dir)
    h, w = _raster_dims(raster_path)
    n_tiles = _count_tiles(h, w, tile_size, overlap)
    out = output_dir / "triton_prob.tif"

    client = TritonSegClient(url=triton_url)
    batch_latencies: list[float] = []

    tracemalloc.start()
    t0 = time.perf_counter()

    with rasterio.open(raster_path) as src:
        profile = src.profile.copy()
        prob_map = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)

        windows = _tile_windows(h, w, tile_size, overlap)

        for coords, batch_array in _generate_tile_batches(
            src, windows, tile_size, batch_size, use_indices=True
        ):
            bt0 = time.perf_counter()
            probs = client.infer_batch(batch_array)
            batch_latencies.append((time.perf_counter() - bt0) * 1000)

            for (row, col), tile_prob in zip(coords, probs):
                prob_map[row : row + tile_size, col : col + tile_size] += tile_prob
                count_map[row : row + tile_size, col : col + tile_size] += 1.0

    valid = count_map > 0
    prob_map[valid] /= count_map[valid]

    out_profile = profile.copy()
    out_profile.update(count=1, dtype="float32", compress="deflate")
    with rasterio.open(out, "w", **out_profile) as dst:
        dst.write(prob_map, 1)

    wall = time.perf_counter() - t0
    _, peak_tm = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    client.close()

    return BenchmarkResult(
        mode="triton",
        raster_size=(h, w),
        tile_size=tile_size,
        overlap=overlap,
        batch_size=batch_size,
        n_tiles=n_tiles,
        wall_time_s=wall,
        tiles_per_sec=n_tiles / wall if wall > 0 else 0,
        peak_rss_mb=_measure_rss_mb(),
        peak_tracemalloc_mb=peak_tm / (1024 * 1024),
        batch_latencies_ms=batch_latencies,
    )


def benchmark_triton_memopt(
    raster_path: Path,
    triton_url: str,
    output_dir: Path,
    tile_size: int,
    overlap: int,
    batch_size: int,
    max_memory_mb: float = 512.0,
) -> BenchmarkResult:
    """Benchmark memory-optimized Triton inference."""
    from services.serving.memory_optimized import memory_optimized_inference

    raster_path, output_dir = Path(raster_path), Path(output_dir)
    h, w = _raster_dims(raster_path)
    n_tiles = _count_tiles(h, w, tile_size, overlap)
    out = output_dir / "triton_memopt_prob.tif"

    tracemalloc.start()
    t0 = time.perf_counter()

    memory_optimized_inference(
        raster_path=raster_path,
        output_path=out,
        triton_url=triton_url,
        tile_size=tile_size,
        overlap=overlap,
        batch_size=batch_size,
        max_memory_mb=max_memory_mb,
    )

    wall = time.perf_counter() - t0
    _, peak_tm = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return BenchmarkResult(
        mode="triton_memopt",
        raster_size=(h, w),
        tile_size=tile_size,
        overlap=overlap,
        batch_size=batch_size,
        n_tiles=n_tiles,
        wall_time_s=wall,
        tiles_per_sec=n_tiles / wall if wall > 0 else 0,
        peak_rss_mb=_measure_rss_mb(),
        peak_tracemalloc_mb=peak_tm / (1024 * 1024),
    )


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
    parser.add_argument("--checkpoint", default="checkpoints/best.ckpt")
    parser.add_argument("--triton-url", default="localhost:8001")
    parser.add_argument(
        "--modes",
        default="baseline,triton,triton_memopt",
        help="Comma-separated list of modes",
    )
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument(
        "--batch-sizes", default="1,8,16,32", help="Comma-separated batch sizes"
    )
    parser.add_argument(
        "--overlaps", default="0,32,64", help="Comma-separated overlap values"
    )
    parser.add_argument(
        "--raster-sizes",
        default="1024,4096,11000",
        help="Comma-separated synthetic raster edge lengths (used when --raster is omitted)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="benchmarks/results")
    parser.add_argument("--max-memory-mb", type=float, default=512.0)

    args = parser.parse_args()
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
                raster_paths[size] = create_synthetic_raster(size, size, tmpdir=tmpdir)

        work_dir = Path(tmpdir) / "outputs"
        work_dir.mkdir()

        for size_key, raster_path in raster_paths.items():
            for ov in overlaps:
                for bs in batch_sizes:
                    for mode in modes:
                        label = f"{mode}|size={size_key}|ov={ov}|bs={bs}"
                        logger.info("Running: %s", label)
                        gpu_before = _gpu_utilization()

                        try:
                            if mode == "baseline":
                                result = benchmark_baseline(
                                    raster_path,
                                    args.checkpoint,
                                    work_dir,
                                    args.tile_size,
                                    ov,
                                    bs,
                                    args.device,
                                )
                            elif mode == "triton":
                                result = benchmark_triton(
                                    raster_path,
                                    args.triton_url,
                                    work_dir,
                                    args.tile_size,
                                    ov,
                                    bs,
                                )
                            elif mode == "triton_memopt":
                                result = benchmark_triton_memopt(
                                    raster_path,
                                    args.triton_url,
                                    work_dir,
                                    args.tile_size,
                                    ov,
                                    bs,
                                    max_memory_mb=args.max_memory_mb,
                                )
                            else:
                                logger.warning("Unknown mode: %s", mode)
                                continue

                            all_results.append(result)
                            logger.info(
                                "  → %.2fs, %.1f tiles/s, %.0f MB RSS",
                                result.wall_time_s,
                                result.tiles_per_sec,
                                result.peak_rss_mb,
                            )
                        except Exception:
                            logger.exception("  → FAILED: %s", label)

    _print_table(all_results)

    results_file = output_dir / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump([r.to_dict() for r in all_results], f, indent=2)
    logger.info("Results saved → %s", results_file)


if __name__ == "__main__":
    main()
