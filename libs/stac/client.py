"""STAC discovery and download for Sentinel-2 imagery."""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import planetary_computer as pc
import pystac_client
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

from libs.geo.bandspec import BandSpec, write_bandspec
from libs.geo.masks import apply_cloud_mask

logger = logging.getLogger(__name__)

os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
os.environ.setdefault("GDAL_INGESTED_BYTES_AT_OPEN", "32768")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff")

STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

SENTINEL2_COLLECTION = "sentinel-2-l2a"

S2_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
S2_BAND_NAMES = ("blue", "green", "red", "nir", "swir16", "swir22")
# L2A digital numbers per reflectance unit; the per-band scale/offset from STAC take precedence
S2_REFLECTANCE_SCALE = 10_000.0
DEFAULT_SCALE_OFFSET = (1e-4, 0.0)


@dataclass
class SceneMetadata:
    scene_id: str
    sensor: str
    datetime: str
    bbox: list[float]
    cloud_cover: float | None
    assets: dict[str, str]
    scales: dict[str, tuple[float, float]] = field(default_factory=dict)


def get_stac_client() -> pystac_client.Client:
    return pystac_client.Client.open(STAC_API_URL, modifier=pc.sign_inplace)


def search_sentinel2(
    bbox: list[float],
    date_range: str,
    max_cloud_cover: float = 20.0,
    limit: int = 50,
) -> list[SceneMetadata]:
    """Search for Sentinel-2 scenes over an AOI and time range.

    Args:
        bbox: [west, south, east, north] in EPSG:4326.
        date_range: ISO 8601 interval, e.g. "2024-01-01/2024-03-31".
        max_cloud_cover: Max cloud cover percentage.
        limit: Maximum number of results.
    """
    client = get_stac_client()
    search = client.search(
        collections=[SENTINEL2_COLLECTION],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
        max_items=limit,
    )
    results = []
    for item in search.items():
        assets = {}
        scales: dict[str, tuple[float, float]] = {}
        for band in S2_BANDS:
            if band in item.assets:
                assets[band] = item.assets[band].href
                rb = (item.assets[band].extra_fields.get("raster:bands") or [{}])[0]
                scales[band] = (
                    float(rb.get("scale", DEFAULT_SCALE_OFFSET[0])),
                    float(rb.get("offset", DEFAULT_SCALE_OFFSET[1])),
                )
        if item.assets.get("SCL"):
            assets["SCL"] = item.assets["SCL"].href

        results.append(
            SceneMetadata(
                scene_id=item.id,
                sensor="sentinel-2",
                datetime=item.datetime.isoformat() if item.datetime else "",
                bbox=list(item.bbox) if item.bbox else bbox,
                cloud_cover=item.properties.get("eo:cloud_cover"),
                assets=assets,
                scales=scales,
            )
        )
    logger.info("Found %d Sentinel-2 scenes for bbox=%s", len(results), bbox)
    return results


def _bbox_to_native(
    target_bbox: list[float], src_crs: rasterio.crs.CRS
) -> tuple[float, float, float, float]:
    """Reproject bbox from EPSG:4326 to the raster's native CRS."""
    west, south, east, north = target_bbox
    if src_crs and src_crs.to_epsg() != 4326:
        transformer = Transformer.from_crs("EPSG:4326", src_crs, always_xy=True)
        west, south = transformer.transform(west, south)
        east, north = transformer.transform(east, north)
    return west, south, east, north


def _read_band(
    url: str,
    target_bbox: list[float],
    out_shape: tuple[int, int],
    resampling: Resampling = Resampling.bilinear,
) -> np.ndarray:
    """Read a single band crop from a remote COG, computing its own window."""
    with rasterio.open(url) as src:
        west, south, east, north = _bbox_to_native(target_bbox, src.crs)
        rb = src.bounds
        west, south = max(west, rb.left), max(south, rb.bottom)
        east, north = min(east, rb.right), min(north, rb.top)
        if west >= east or south >= north:
            raise ValueError("No spatial overlap")
        window = from_bounds(west, south, east, north, transform=src.transform)
        window = window.round_offsets().round_lengths()
        if window.width < 1 or window.height < 1:
            raise ValueError("Zero-pixel window")
        return src.read(1, window=window, out_shape=out_shape, resampling=resampling)


def assemble_crop(
    band_arrays: dict[str, np.ndarray],
    scl: np.ndarray,
    scales: dict[str, tuple[float, float]],
) -> np.ndarray:
    """DN bands -> cloud-masked float32 reflectance cube (C,H,W) in S2_BAND_NAMES order.

    Reflectance = dn * scale + offset per band (fallback (1e-4, 0.0)); pixels that the SCL
    marks as not clear are NaN in every band.
    """
    missing = [b for b in S2_BANDS if b not in band_arrays]
    if missing:
        raise ValueError(f"missing bands {missing}")
    layers = []
    for band in S2_BANDS:
        scale, offset = scales.get(band, DEFAULT_SCALE_OFFSET)
        layers.append(band_arrays[band].astype(np.float32) * scale + offset)
    return apply_cloud_mask(np.stack(layers), scl)


def download_scene_crop(
    scene: SceneMetadata,
    target_bbox: list[float],
    output_dir: Path,
    bands: list[str] | None = None,
    max_workers: int = 6,
) -> Path:
    """Download a cropped region of a scene's bands and stack into a single GeoTIFF.

    Band reads are parallelized across threads for much faster I/O.
    Each band computes its own window from the bbox so mixed-resolution
    bands (10m vs 20m) are handled correctly and resampled to a common grid.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{scene.scene_id}.tif"

    if out_path.exists():
        logger.debug("Scene %s already downloaded at %s", scene.scene_id, out_path)
        return out_path

    band_keys = [k for k in (bands or S2_BANDS) if k in scene.assets and k != "SCL"]
    missing = [k for k in S2_BANDS if k not in band_keys]
    if missing or "SCL" not in scene.assets:
        raise ValueError(f"Scene {scene.scene_id} lacks bands {missing} or SCL")
    urls = [(key, scene.assets[key]) for key in S2_BANDS]

    # Use the first (10m) band to establish the output grid
    first_url = urls[0][1]
    with rasterio.open(first_url) as src:
        src_crs = src.crs
        west, south, east, north = _bbox_to_native(target_bbox, src_crs)
        rb = src.bounds
        west, south = max(west, rb.left), max(south, rb.bottom)
        east, north = min(east, rb.right), min(north, rb.top)
        window = from_bounds(west, south, east, north, transform=src.transform)
        window = window.round_offsets().round_lengths()
        ref_transform = rasterio.windows.transform(window, src.transform)
        ref_shape = (int(window.height), int(window.width))

    profile = {
        "driver": "GTiff",
        "height": ref_shape[0],
        "width": ref_shape[1],
        "count": len(S2_BANDS),
        "dtype": "float32",
        "nodata": float("nan"),
        "transform": ref_transform,
        "compress": "deflate",
        "crs": src_crs,
    }

    arrays: dict[str, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(urls) + 1)) as pool:
        futures = {pool.submit(_read_band, url, target_bbox, ref_shape): key for key, url in urls}
        futures[
            pool.submit(
                _read_band,
                scene.assets["SCL"],
                target_bbox,
                ref_shape,
                Resampling.nearest,
            )
        ] = "SCL"
        for future in as_completed(futures):
            # A missing band or SCL makes the crop unusable: let the error propagate.
            arrays[futures[future]] = future.result()

    scl = arrays.pop("SCL")
    cube = assemble_crop(arrays, scl, scene.scales)

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(cube)
        write_bandspec(dst, BandSpec(names=S2_BAND_NAMES, reflectance_scale=1.0))

    logger.info("Downloaded %s → %s (%d bands)", scene.scene_id, out_path, len(S2_BANDS))
    return out_path


def _download_one_scene(
    scene: SceneMetadata,
    target_bbox: list[float],
    output_dir: Path,
    bands: list[str] | None,
    idx: int,
    total: int,
) -> Path | None:
    t0 = time.time()
    try:
        path = download_scene_crop(scene, target_bbox, output_dir, bands=bands)
        sz = path.stat().st_size / 1e6
        logger.info(
            "  [%d/%d] %s  %.1f MB  %.1fs",
            idx,
            total,
            scene.scene_id,
            sz,
            time.time() - t0,
        )
        return path
    except Exception as exc:
        logger.warning("  [%d/%d] SKIP %s: %s", idx, total, scene.scene_id, exc)
        return None


def ingest_aoi(
    bbox: list[float],
    date_range: str,
    output_dir: Path,
    max_cloud_cover: float = 20.0,
    limit: int = 50,
    max_scene_workers: int = 4,
) -> list[Path]:
    """Search Sentinel-2 L2A over the AOI and download cropped scenes concurrently."""
    output_dir = Path(output_dir)
    s2_dir = output_dir / "sentinel2"
    scenes = search_sentinel2(bbox, date_range, max_cloud_cover, limit=limit)
    downloaded: list[Path] = []
    with ThreadPoolExecutor(max_workers=max_scene_workers) as pool:
        futures = {
            pool.submit(_download_one_scene, scene, bbox, s2_dir, S2_BANDS, i, len(scenes)): scene
            for i, scene in enumerate(scenes, 1)
        }
        for future in as_completed(futures):
            path = future.result()
            if path is not None:
                downloaded.append(path)
    logger.info("Ingested %d scenes for AOI bbox=%s", len(downloaded), bbox)
    return downloaded
