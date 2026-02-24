"""STAC discovery and download for Sentinel-2 and Sentinel-1 imagery."""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import planetary_computer as pc
import pystac_client
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.windows import from_bounds
from shapely.geometry import box, mapping

from libs.config import settings

logger = logging.getLogger(__name__)

os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
os.environ.setdefault("GDAL_INGESTED_BYTES_AT_OPEN", "32768")
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff")

STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

SENTINEL2_COLLECTION = "sentinel-2-l2a"
SENTINEL1_COLLECTION = "sentinel-1-rtc"

S2_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
S2_BAND_NAMES = ["blue", "green", "red", "nir", "swir16", "swir22"]


@dataclass
class SceneMetadata:
    scene_id: str
    sensor: str
    datetime: str
    bbox: list[float]
    cloud_cover: float | None
    assets: dict[str, str]


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
        for band in S2_BANDS:
            if band in item.assets:
                assets[band] = item.assets[band].href
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
            )
        )
    logger.info("Found %d Sentinel-2 scenes for bbox=%s", len(results), bbox)
    return results


def search_sentinel1(
    bbox: list[float],
    date_range: str,
    limit: int = 50,
) -> list[SceneMetadata]:
    """Search for Sentinel-1 RTC scenes."""
    client = get_stac_client()
    search = client.search(
        collections=[SENTINEL1_COLLECTION],
        bbox=bbox,
        datetime=date_range,
        max_items=limit,
    )
    results = []
    for item in search.items():
        assets = {}
        for key in ("vh", "vv"):
            if key in item.assets:
                assets[key] = item.assets[key].href

        results.append(
            SceneMetadata(
                scene_id=item.id,
                sensor="sentinel-1",
                datetime=item.datetime.isoformat() if item.datetime else "",
                bbox=list(item.bbox) if item.bbox else bbox,
                cloud_cover=None,
                assets=assets,
            )
        )
    logger.info("Found %d Sentinel-1 scenes for bbox=%s", len(results), bbox)
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
    url: str, target_bbox: list[float], out_shape: tuple[int, int]
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
        return src.read(1, window=window, out_shape=out_shape)


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

    asset_keys = bands or list(scene.assets.keys())
    urls = [(key, scene.assets[key]) for key in asset_keys if key in scene.assets]
    if not urls:
        raise ValueError(f"No band URLs for scene {scene.scene_id}")

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
        profile = src.profile.copy()

    profile.update(
        driver="GTiff",
        height=ref_shape[0],
        width=ref_shape[1],
        count=len(urls),
        transform=ref_transform,
        compress="deflate",
        crs=src_crs,
    )

    band_arrays: list[tuple[int, np.ndarray]] = []

    with ThreadPoolExecutor(max_workers=min(max_workers, len(urls))) as pool:
        futures = {
            pool.submit(_read_band, url, target_bbox, ref_shape): (idx, key)
            for idx, (key, url) in enumerate(urls)
        }
        for future in as_completed(futures):
            idx, key = futures[future]
            try:
                band_arrays.append((idx, future.result()))
            except Exception as exc:
                logger.warning("Failed to read %s band %s: %s", scene.scene_id, key, exc)

    if not band_arrays:
        raise ValueError(f"No bands downloaded for scene {scene.scene_id}")

    band_arrays.sort(key=lambda x: x[0])
    profile["count"] = len(band_arrays)

    with rasterio.open(out_path, "w", **profile) as dst:
        for i, (_, arr) in enumerate(band_arrays, 1):
            dst.write(arr, i)

    logger.info("Downloaded %s → %s (%d bands)", scene.scene_id, out_path, len(band_arrays))
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
            idx, total, scene.scene_id, sz, time.time() - t0,
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
    include_s1: bool = False,
    limit: int = 50,
    max_scene_workers: int = 4,
) -> list[Path]:
    """End-to-end ingestion for a single AOI: search + download crops.

    Scene downloads run concurrently (max_scene_workers threads),
    and each scene's band reads are also threaded internally.
    """
    output_dir = Path(output_dir)
    s2_dir = output_dir / "sentinel2"
    downloaded: list[Path] = []

    scenes = search_sentinel2(bbox, date_range, max_cloud_cover, limit=limit)

    with ThreadPoolExecutor(max_workers=max_scene_workers) as pool:
        futures = {
            pool.submit(
                _download_one_scene, scene, bbox, s2_dir, S2_BANDS, i, len(scenes)
            ): scene
            for i, scene in enumerate(scenes, 1)
        }
        for future in as_completed(futures):
            path = future.result()
            if path is not None:
                downloaded.append(path)

    if include_s1:
        s1_dir = output_dir / "sentinel1"
        s1_scenes = search_sentinel1(bbox, date_range, limit=limit)
        with ThreadPoolExecutor(max_workers=max_scene_workers) as pool:
            futures = {
                pool.submit(
                    _download_one_scene, scene, bbox, s1_dir, None, i, len(s1_scenes)
                ): scene
                for i, scene in enumerate(s1_scenes, 1)
            }
            for future in as_completed(futures):
                path = future.result()
                if path is not None:
                    downloaded.append(path)

    logger.info("Ingested %d scenes for AOI bbox=%s", len(downloaded), bbox)
    return downloaded
