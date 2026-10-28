"""Sliding-window inference over a composite: windows -> predictor -> averaged probability raster.

Edge handling: windows are clamped so the last row/column of windows ends exactly at the raster
edge (every pixel is covered); rasters smaller than one tile are padded by reflection.

Memory: ``Accumulator(memmap=True)`` keeps the two full-size maps (sum and count) on disk and only
one batch of tiles is in RAM. Because ``iter_windows`` walks row-major and ``np.memmap`` pages
lazily, no explicit strip loop is needed while accumulating; finalising, nodata masking and
writing go in row strips, so the probability map is never copied into RAM.

Nodata: pixels whose first composite band is NaN (outside the swath, never clear) are NaN in the
probability raster, so the model's output on zero-filled input never becomes a detection.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from libs.features.indices import add_indices, check_stats_channels, normalize
from libs.geo.bandspec import read_bandspec
from libs.inference.predictors import Predictor

logger = logging.getLogger(__name__)
STRIP_ROWS = 1024  # rows per strip when finalising, masking and writing full-size maps


def iter_windows(height: int, width: int, tile: int, overlap: int) -> Iterator[tuple[int, int]]:
    """Top-left (row, col) of every window; the last window in each axis is clamped to the edge."""
    step = tile - overlap
    if step <= 0:
        raise ValueError("overlap must be smaller than tile")

    def starts(n: int) -> list[int]:
        if n <= tile:
            return [0]
        s = list(range(0, n - tile, step))
        s.append(n - tile)
        return sorted(set(s))

    for r in starts(height):
        for c in starts(width):
            yield r, c


def _strips(height: int) -> Iterator[tuple[int, int]]:
    for r0 in range(0, height, STRIP_ROWS):
        yield r0, min(r0 + STRIP_ROWS, height)


class Accumulator:
    """Sum of probabilities and counts per pixel; finalize() leaves NaN where count is 0."""

    def __init__(self, shape: tuple[int, int], memmap: bool = False, tmpdir: Path | None = None):
        self.shape = shape
        self._tmp = None
        self._finalized = False
        if memmap:
            self._tmp = tempfile.TemporaryDirectory(dir=tmpdir, prefix="slidingwindow_")
            base = Path(self._tmp.name)
            self.prob = np.memmap(base / "prob.dat", dtype=np.float32, mode="w+", shape=shape)
            self.count = np.memmap(base / "count.dat", dtype=np.float32, mode="w+", shape=shape)
            self.prob[:] = 0.0
            self.count[:] = 0.0
        else:
            self.prob = np.zeros(shape, dtype=np.float32)
            self.count = np.zeros(shape, dtype=np.float32)

    def add(self, row: int, col: int, tile_prob: np.ndarray) -> None:
        h, w = tile_prob.shape
        self.prob[row : row + h, col : col + w] += tile_prob
        self.count[row : row + h, col : col + w] += 1.0

    def finalize(self) -> np.ndarray:
        """Mean probability per pixel, computed in place (memmap-backed in memmap mode).

        Idempotent; the returned array is only valid until close().
        """
        if not self._finalized:
            for r0, r1 in _strips(self.shape[0]):
                p, c = self.prob[r0:r1], self.count[r0:r1]
                np.divide(p, c, out=p, where=c > 0)
                p[c == 0] = np.nan
            self._finalized = True
        return self.prob

    def close(self) -> None:
        if self._tmp is not None:
            del self.prob, self.count
            self._tmp.cleanup()
            self._tmp = None


def read_tile(
    src: rasterio.io.DatasetReader, row: int, col: int, tile: int, names, stats: dict
) -> np.ndarray:
    """Read one window and preprocess exactly like the training dataset does."""
    h = min(tile, src.height - row)
    w = min(tile, src.width - col)
    data = src.read(window=Window(col, row, w, h)).astype(np.float32)
    if h < tile or w < tile:  # raster smaller than a tile: reflect-pad
        data = np.pad(data, ((0, 0), (0, tile - h), (0, tile - w)), mode="reflect")
    data, channels = add_indices(data, names)
    check_stats_channels(channels, stats)
    data = normalize(data, stats)
    return np.nan_to_num(data, nan=0.0)


def run_sliding_window(
    raster_path: Path,
    output_path: Path,
    predictor: Predictor,
    stats: dict,
    tile: int = 256,
    overlap: int = 64,
    batch_size: int = 16,
    memmap: bool = False,
    tmpdir: Path | None = None,
) -> Path:
    """Predict a whole composite and write a single-band float32 probability GeoTIFF."""
    raster_path, output_path = Path(raster_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(raster_path) as src:
        names = read_bandspec(src).names
        h, w = src.height, src.width
        acc = Accumulator((h, w), memmap=memmap, tmpdir=tmpdir)
        try:
            batch, coords = [], []
            n = 0
            for r, c in iter_windows(h, w, tile, overlap):
                batch.append(read_tile(src, r, c, tile, names, stats))
                coords.append((r, c))
                if len(batch) == batch_size:
                    n += _flush(predictor, batch, coords, acc, h, w)
                    batch, coords = [], []
            if batch:
                n += _flush(predictor, batch, coords, acc, h, w)
            prob = acc.finalize()
            _mask_nodata(src, prob)
            _write_prob(output_path, prob, src.profile)
        finally:
            acc.close()
    logger.info("Predicted %s -> %s (%d windows)", raster_path.name, output_path.name, n)
    return output_path


def _mask_nodata(src: rasterio.io.DatasetReader, prob: np.ndarray) -> None:
    """NaN wherever the composite's first band is NaN (in place, strip by strip)."""
    for r0, r1 in _strips(src.height):
        band = src.read(1, window=Window(0, r0, src.width, r1 - r0))
        prob[r0:r1][np.isnan(band)] = np.nan


def _write_prob(output_path: Path, prob: np.ndarray, src_profile: dict) -> None:
    profile = dict(src_profile)
    profile.update(count=1, dtype="float32", nodata=float("nan"), compress="deflate")
    # rasterio profiles do not carry tags, but older writers might; never copy a stale BANDSPEC
    profile.pop("tags", None)
    for key in ("blockxsize", "blockysize", "tiled"):  # stale layout from the source
        profile.pop(key, None)
    height, width = prob.shape
    with rasterio.open(output_path, "w", **profile) as dst:
        for r0, r1 in _strips(height):
            dst.write(prob[r0:r1], 1, window=Window(0, r0, width, r1 - r0))


def _flush(
    predictor: Predictor,
    batch: list[np.ndarray],
    coords: list[tuple[int, int]],
    acc: Accumulator,
    h: int,
    w: int,
) -> int:
    probs = predictor.predict(np.stack(batch))
    for (r, c), p in zip(coords, probs, strict=True):
        acc.add(r, c, p[: h - r, : w - c])  # crop the reflect-padding for small rasters
    return len(batch)
