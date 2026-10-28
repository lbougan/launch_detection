"""Postprocessing: threshold, connected components, polygonize, rank, dedupe."""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import rasterio
import shapely
from pyproj import Transformer
from rasterio.features import shapes as rasterio_shapes
from rasterio.warp import transform as warp_transform
from scipy.ndimage import label as scipy_label
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN

from libs.geo.projection import local_aeqd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    """A candidate site. `prob` is the model's calibrated-ish confidence; `rank_score` orders.

    `geometry` is in EPSG:4326 (lon/lat); `area_km2` and `compactness` were computed in the
    probability raster's projected CRS.
    """

    geometry: Polygon | None
    prob: float
    rank_score: float | None
    area_km2: float
    compactness: float
    centroid_lon: float
    centroid_lat: float
    evidence: dict


def threshold_and_extract(
    prob_raster_path: str,
    threshold: float = 0.5,
    min_area_px: int = 50,
) -> list[Detection]:
    """Convert probability raster → ranked candidate polygons.

    Pipeline:
      1. Threshold to binary
      2. Connected components
      3. Polygonize each component
      4. Compute geometric features (in the raster's projected CRS)
      5. Reproject each polygon to EPSG:4326 and return detections sorted by prob
    """
    with rasterio.open(prob_raster_path) as src:
        prob = src.read(1)
        transform = src.transform
        crs = src.crs
        if crs is None or not crs.is_projected:
            raise ValueError(f"{prob_raster_path} must be in a projected (metric) CRS, got {crs}")
        pixel_area_m2 = abs(transform.a * transform.e)
    to_lonlat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform

    binary = (prob >= threshold).astype(np.uint8)
    labeled, n_components = scipy_label(binary)
    logger.info("Found %d connected components at threshold=%.2f", n_components, threshold)

    detections = []
    for component_id in range(1, n_components + 1):
        component_mask = (labeled == component_id).astype(np.uint8)
        area_px = component_mask.sum()

        if area_px < min_area_px:
            continue

        component_probs = prob[labeled == component_id]
        mean_prob = float(np.mean(component_probs))

        polygons = []
        for geom, value in rasterio_shapes(component_mask, transform=transform):
            if value == 1:
                polygons.append(shape(geom))

        if not polygons:
            continue

        merged = unary_union(polygons)
        if isinstance(merged, MultiPolygon):
            merged = max(merged.geoms, key=lambda g: g.area)

        area_km2 = area_px * pixel_area_m2 / 1e6
        compactness = _compactness(merged)  # metric, before reprojection
        centroid = merged.centroid
        lon, lat = to_lonlat(centroid.x, centroid.y)

        detections.append(
            Detection(
                geometry=shapely.transform(merged, to_lonlat, interleaved=False),
                prob=mean_prob,
                rank_score=None,
                area_km2=area_km2,
                compactness=compactness,
                centroid_lon=float(lon),
                centroid_lat=float(lat),
                evidence={"mean_prob": mean_prob, "area_px": int(area_px)},
            )
        )

    detections.sort(key=lambda d: d.prob, reverse=True)
    logger.info("Extracted %d candidate detections (min_area=%dpx)", len(detections), min_area_px)
    return detections


def _compactness(polygon: Polygon) -> float:
    """Polsby-Popper compactness: 4*pi*area / perimeter^2. Circle = 1."""
    if polygon.length == 0:
        return 0.0
    return (4 * np.pi * polygon.area) / (polygon.length**2)


def deduplicate_detections(
    detections: list[Detection],
    method: str = "dbscan",
    eps_m: float = 2000.0,
    iou_threshold: float = 0.3,
) -> list[Detection]:
    """Merge overlapping/nearby detections.

    Args:
        detections: Candidate detections (already scored).
        method: "dbscan" (cluster by centroid distance) or "iou" (merge by overlap).
        eps_m: DBSCAN epsilon in metres, applied in a local azimuthal equidistant frame.
        iou_threshold: IoU threshold for merging (only for "iou" method).

    Returns:
        Deduplicated detections, one per cluster (highest rank_score kept).
    """
    if not detections:
        return []

    if method == "dbscan":
        return _dedupe_dbscan(detections, eps_m)
    elif method == "iou":
        return _dedupe_iou(detections, iou_threshold)
    else:
        raise ValueError(f"Unknown dedupe method: {method}")


def _rank_key(d: Detection) -> float:
    return d.rank_score if d.rank_score is not None else d.prob


def _dedupe_dbscan(detections: list[Detection], eps_m: float) -> list[Detection]:
    """Cluster centroids in metres (local AEQD centred on the mean; fine for one AOI)."""
    lons = np.array([d.centroid_lon for d in detections])
    lats = np.array([d.centroid_lat for d in detections])
    crs = local_aeqd(float(lons.mean()), float(lats.mean()))
    xs, ys = warp_transform("EPSG:4326", crs, lons.tolist(), lats.tolist())
    coords = np.column_stack([xs, ys])
    clustering = DBSCAN(eps=eps_m, min_samples=1, metric="euclidean").fit(coords)

    cluster_best: dict[int, Detection] = {}
    for det, label in zip(detections, clustering.labels_, strict=True):
        if label not in cluster_best or _rank_key(det) > _rank_key(cluster_best[label]):
            cluster_best[label] = det

    result = sorted(cluster_best.values(), key=_rank_key, reverse=True)
    logger.info("DBSCAN dedupe: %d -> %d detections", len(detections), len(result))
    return result


def _dedupe_iou(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    """Greedy IoU-based non-maximum suppression."""
    remaining = list(detections)
    kept = []

    while remaining:
        best = remaining.pop(0)
        kept.append(best)
        remaining = [d for d in remaining if _iou(best.geometry, d.geometry) < iou_threshold]

    logger.info("IoU dedupe: %d → %d detections", len(detections), len(kept))
    return kept


def _iou(a: Polygon, b: Polygon) -> float:
    if not a.intersects(b):
        return 0.0
    intersection = a.intersection(b).area
    union = a.area + b.area - intersection
    return intersection / union if union > 0 else 0.0


def score_detections(
    detections: list[Detection],
    prob_weight: float = 0.6,
    compactness_weight: float = 0.2,
    area_weight: float = 0.2,
    ideal_area_km2: float = 2.0,
) -> list[Detection]:
    """Return new detections with a composite `rank_score` (`prob` is left untouched).

    Combines model probability, shape compactness, and area prior.
    """
    scored = []
    for det in detections:
        area_score = 1.0 - min(abs(det.area_km2 - ideal_area_km2) / ideal_area_km2, 1.0)
        composite = (
            prob_weight * det.prob + compactness_weight * det.compactness + area_weight * area_score
        )
        breakdown = {
            "prob": det.prob,
            "compactness": det.compactness,
            "area_score": area_score,
        }
        scored.append(
            dataclasses.replace(
                det,
                rank_score=composite,
                evidence={**det.evidence, "rank_breakdown": breakdown},
            )
        )

    return sorted(scored, key=lambda d: d.rank_score, reverse=True)


def detections_to_geodataframe(detections: list[Detection]) -> gpd.GeoDataFrame:
    """Convert detections to a GeoDataFrame for storage or visualization."""
    if not detections:
        return gpd.GeoDataFrame(
            columns=["geometry", "prob", "rank_score", "area_km2", "compactness", "rank_breakdown"]
        )

    return gpd.GeoDataFrame(
        [
            {
                "geometry": d.geometry,
                "prob": d.prob,
                "rank_score": d.rank_score,
                "area_km2": d.area_km2,
                "compactness": d.compactness,
                "centroid_lon": d.centroid_lon,
                "centroid_lat": d.centroid_lat,
                "rank_breakdown": d.evidence.get("rank_breakdown", {}),
            }
            for d in detections
        ],
        crs="EPSG:4326",
    )
