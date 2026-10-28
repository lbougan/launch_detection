"""STAC discovery and download for Sentinel-2 imagery."""

from __future__ import annotations

import logging
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import planetary_computer as pc
import pystac_client
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds

from libs.geo.bandspec import S2_BAND_NAMES, TAG_KEY, BandSpec, write_bandspec
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
DEFAULT_SCALE_OFFSET = (1e-4, 0.0)
# ESA added a +1000 DN offset from processing baseline 04.00 (25 Jan 2022)
BASELINE_OFFSET_FROM = 4.0
BASELINE_OFFSET = -0.1
__all__ = ["S2_BAND_NAMES"]  # re-exported from libs.geo.bandspec


TARGET_RES_M = 10.0


@dataclass(frozen=True)
class Grid:
    """The one target grid of an AOI: every scene crop is warped onto it."""

    crs: CRS
    transform: Affine
    width: int
    height: int


def utm_crs(lon: float, lat: float) -> CRS:
    """WGS84 / UTM zone of (lon, lat): EPSG 326xx north, 327xx south."""
    zone = min(max(int((lon + 180.0) // 6.0) + 1, 1), 60)
    return CRS.from_user_input(f"EPSG:{(32600 if lat >= 0 else 32700) + zone}")


def aoi_target_grid(bbox: list[float], res: float = TARGET_RES_M) -> Grid:
    """UTM grid (zone of the AOI centre) covering the lon/lat bbox, snapped to `res` metres."""
    west, south, east, north = bbox
    crs = utm_crs((west + east) / 2.0, (south + north) / 2.0)
    left, bottom, right, top = _bbox_to_native(bbox, crs)
    x0, y0 = math.floor(left / res) * res, math.ceil(top / res) * res
    width = math.ceil((right - x0) / res)
    height = math.ceil((y0 - bottom) / res)
    return Grid(crs=crs, transform=from_origin(x0, y0, res, res), width=width, height=height)


def crop_matches_grid(path: Path, grid: Grid) -> bool:
    with rasterio.open(path) as src:
        return (src.crs, src.transform, src.width, src.height) == (
            grid.crs,
            grid.transform,
            grid.width,
            grid.height,
        )


@dataclass
class SceneMetadata:
    scene_id: str
    sensor: str
    datetime: str
    bbox: list[float]
    cloud_cover: float | None
    assets: dict[str, str]
    scales: dict[str, tuple[float, float]] = field(default_factory=dict)


def scale_offset_for(asset_extra: dict, properties: dict) -> tuple[float, float]:
    """Reflectance = DN * scale + offset for one asset.

    Explicit `raster:bands` metadata wins; otherwise the offset follows from the
    `s2:processing_baseline` item property (-0.1 from baseline 04.00).
    """
    rb = (asset_extra.get("raster:bands") or [{}])[0]
    if "scale" in rb or "offset" in rb:
        return (
            float(rb.get("scale", DEFAULT_SCALE_OFFSET[0])),
            float(rb.get("offset", DEFAULT_SCALE_OFFSET[1])),
        )
    baseline = float(properties.get("s2:processing_baseline", "0") or 0)
    offset = BASELINE_OFFSET if baseline >= BASELINE_OFFSET_FROM else 0.0
    return (DEFAULT_SCALE_OFFSET[0], offset)


def is_pipeline_crop(path: Path) -> bool:
    """True if the GeoTIFF carries the BANDSPEC tag (v0.1 crops are untagged, unmasked uint16)."""
    with rasterio.open(path) as src:
        return TAG_KEY in src.tags()


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
                scales[band] = scale_offset_for(item.assets[band].extra_fields, item.properties)
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
    """Reproject bbox from EPSG:4326 to the raster's native CRS.

    The edges are densified, so the native box encloses the whole lon/lat box (at high latitude
    its edges bulge in UTM and the two corners alone would clip the crop).
    """
    if not src_crs or src_crs == CRS.from_epsg(4326):
        return tuple(target_bbox)
    return transform_bounds("EPSG:4326", src_crs, *target_bbox, densify_pts=21)


def _read_band(url: str, grid: Grid, resampling: Resampling = Resampling.bilinear) -> np.ndarray:
    """Warp one band of a remote COG onto `grid`; pixels outside the scene come out as 0
    (the L2A nodata value, SCL class 0), which the cloud mask turns into NaN."""
    with (
        rasterio.open(url) as src,
        WarpedVRT(
            src,
            crs=grid.crs,
            transform=grid.transform,
            width=grid.width,
            height=grid.height,
            resampling=resampling,
        ) as vrt,
    ):
        return vrt.read(1)


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
    grid: Grid,
    output_dir: Path,
    max_workers: int = 6,
) -> Path:
    """Warp a scene's bands and SCL onto the AOI grid and write one float32 reflectance GeoTIFF.

    Band reads run in parallel threads. Every crop of an AOI lands on the identical grid, whatever
    the scene's MGRS tile or UTM zone (bilinear for reflectance, nearest for the SCL).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{scene.scene_id}.tif"

    if out_path.exists():
        if is_pipeline_crop(out_path) and crop_matches_grid(out_path, grid):
            logger.debug("Scene %s already downloaded at %s", scene.scene_id, out_path)
            return out_path
        logger.warning(
            "Re-downloading %s: %s is legacy or on another grid", scene.scene_id, out_path
        )

    missing = [k for k in S2_BANDS if k not in scene.assets]
    if missing or "SCL" not in scene.assets:
        raise ValueError(f"Scene {scene.scene_id} lacks bands {missing} or SCL")

    profile = {
        "driver": "GTiff",
        "height": grid.height,
        "width": grid.width,
        "count": len(S2_BANDS),
        "dtype": "float32",
        "nodata": float("nan"),
        "transform": grid.transform,
        "compress": "deflate",
        "crs": grid.crs,
    }

    arrays: dict[str, np.ndarray] = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(S2_BANDS) + 1)) as pool:
        futures = {pool.submit(_read_band, scene.assets[key], grid): key for key in S2_BANDS}
        futures[pool.submit(_read_band, scene.assets["SCL"], grid, Resampling.nearest)] = "SCL"
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
    grid: Grid,
    output_dir: Path,
    idx: int,
    total: int,
) -> Path | None:
    t0 = time.time()
    try:
        path = download_scene_crop(scene, grid, output_dir)
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
    """Search Sentinel-2 L2A over the AOI and warp the scenes onto one AOI grid, concurrently."""
    output_dir = Path(output_dir)
    s2_dir = output_dir / "sentinel2"
    grid = aoi_target_grid(bbox)
    scenes = search_sentinel2(bbox, date_range, max_cloud_cover, limit=limit)
    downloaded: list[Path] = []
    with ThreadPoolExecutor(max_workers=max_scene_workers) as pool:
        futures = {
            pool.submit(_download_one_scene, scene, grid, s2_dir, i, len(scenes)): scene
            for i, scene in enumerate(scenes, 1)
        }
        for future in as_completed(futures):
            path = future.result()
            if path is not None:
                downloaded.append(path)
    logger.info("Ingested %d scenes for AOI bbox=%s", len(downloaded), bbox)
    return downloaded
