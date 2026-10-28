"""Global per-channel statistics over the training composites (decision 4)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio

from libs.features.indices import add_indices
from libs.geo.bandspec import read_bandspec


def compute_channel_stats(
    composite_paths: list[Path], sample_px: int = 2_000_000, seed: int = 0
) -> dict:
    """p2/p98/mean/std per channel (bands + indices) from a random pixel sample.

    mean/std are computed on the clipped-and-scaled values so that `normalize` yields
    zero-mean unit-variance inputs. NaN (cloud/nodata) pixels are excluded.
    """
    rng = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    channels: tuple[str, ...] | None = None
    per_file = max(1, sample_px // max(1, len(composite_paths)))
    for p in composite_paths:
        with rasterio.open(p) as src:
            cube = src.read().astype(np.float32)
            spec = read_bandspec(src)
        cube, names = add_indices(cube, spec.names)
        channels = channels or names
        flat = cube.reshape(cube.shape[0], -1)
        valid = ~np.isnan(flat).any(axis=0)
        cols = np.flatnonzero(valid)
        if cols.size == 0:
            continue
        pick = rng.choice(cols, size=min(per_file, cols.size), replace=False)
        samples.append(flat[:, pick])
    if not samples or channels is None:
        raise ValueError("no valid pixels in the given composites")
    x = np.concatenate(samples, axis=1)  # (C, N)
    p2 = np.percentile(x, 2, axis=1)
    p98 = np.percentile(x, 98, axis=1)
    scaled = np.clip((x - p2[:, None]) / np.maximum(p98 - p2, 1e-6)[:, None], 0.0, 1.0)
    return {
        "channels": list(channels),
        "p2": p2.tolist(),
        "p98": p98.tolist(),
        "mean": scaled.mean(axis=1).tolist(),
        "std": np.maximum(scaled.std(axis=1), 1e-6).tolist(),
        "n_samples": int(x.shape[1]),
    }


def save_stats(stats: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2))


def load_stats(path: Path) -> dict:
    return json.loads(Path(path).read_text())
