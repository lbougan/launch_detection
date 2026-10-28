"""Prefect tasks: atomic units of work in the pipeline."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import rasterio
from prefect import task
from sqlalchemy import create_engine, text

from libs.config import settings
from libs.features.indices import build_temporal_composite
from libs.features.stats import compute_channel_stats, load_stats, save_stats
from libs.geo.bandspec import write_bandspec
from libs.geo.chips import chip_raster, load_known_sites
from libs.geo.labels import build_label_mask, write_label
from libs.geo.postprocess import (
    deduplicate_detections,
    detections_to_geodataframe,
    score_detections,
    threshold_and_extract,
)
from libs.geo.splits import assign_splits
from libs.inference.sliding_window import run_sliding_window
from libs.stac.client import ingest_aoi

TRITON_URL = os.environ.get("TRITON_URL", "")

logger = logging.getLogger(__name__)


@task(name="train-model")
def train_model_task(manifest_path: str, stats_path: str | None = None, **overrides) -> str:
    """Train via the shared `fit` loop (A13); return the best checkpoint path.

    `overrides` are `FitConfig` fields (model_type, encoder, lr, epochs, ...).
    """
    from libs.config import default_stats_path
    from services.training.fit import FitConfig, fit

    overrides.setdefault("mlflow_tracking_uri", settings.mlflow_tracking_uri)
    if overrides.get("gpus", 0) > 0:
        overrides.setdefault("precision", "16-mixed")
    cfg = FitConfig(
        manifest_path=manifest_path,
        stats_path=stats_path or str(default_stats_path()),
        **overrides,
    )
    return fit(cfg)


@task(name="ingest-aoi", retries=2, retry_delay_seconds=60)
def ingest_aoi_task(
    aoi_name: str,
    bbox: list[float],
    date_range: str,
    output_dir: str,
    max_cloud_cover: float = 20.0,
) -> list[str]:
    """Ingest satellite imagery for a single AOI."""
    paths = ingest_aoi(
        bbox=bbox,
        date_range=date_range,
        output_dir=Path(output_dir),
        max_cloud_cover=max_cloud_cover,
    )
    return [str(p) for p in paths]


@task(name="build-composite")
def build_composite_task(
    scene_paths: list[str],
    output_path: str,
    method: str = "median",
) -> str:
    """Build a temporal composite from multiple scenes."""
    paths = [Path(p) for p in scene_paths]
    if not paths:
        raise ValueError("No scenes to composite")

    composite, profile, spec = build_temporal_composite(paths, method)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    profile.update(dtype="float32", nodata=float("nan"), compress="deflate")
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(composite)
        write_bandspec(dst, spec)

    logger.info("Built composite → %s", out)
    return str(out)


@task(name="compute-stats")
def compute_stats_task(composite_paths: list[str], output_path: str) -> str:
    """Compute global per-channel normalisation stats over the given composites."""
    stats = compute_channel_stats([Path(p) for p in composite_paths])
    save_stats(stats, Path(output_path))
    logger.info("Wrote channel stats (%d samples) -> %s", stats["n_samples"], output_path)
    return output_path


@task(name="tile-composite")
def tile_composite_task(
    composite_path: str,
    output_dir: str,
    tile_size: int = 256,
    overlap: int = 32,
) -> list[dict]:
    """Chip a composite raster into ML-ready tiles."""
    chips = chip_raster(
        raster_path=Path(composite_path),
        output_dir=Path(output_dir),
        tile_size=tile_size,
        overlap=overlap,
    )
    return chips


@task(name="assign-splits")
def assign_splits_task(
    chip_manifest: list[dict],
    known_sites_path: str,
    splits_path: str,
) -> list[dict]:
    """Assign site-held-out train/val/test splits to chips."""
    sites = load_known_sites(Path(known_sites_path))
    splits = json.loads(Path(splits_path).read_text())
    return assign_splits(chip_manifest, sites, splits, exclusion_km=float(splits["exclusion_km"]))


@task(name="build-labels")
def build_labels_task(
    chip_manifest: list[dict],
    known_sites_path: str,
) -> list[dict]:
    """Generate weak label masks for chips (splits are assigned upstream)."""
    sites = load_known_sites(Path(known_sites_path))
    enriched = [dict(chip) for chip in chip_manifest]

    for chip in enriched:
        with rasterio.open(chip["path"]) as src:
            mask = build_label_mask(src, sites)
            label_path = Path(chip["path"]).parent / f"{chip['tile_id']}_label.tif"
            chip["label_path"] = str(write_label(label_path, mask, src))

    return enriched


@task(name="run-inference")
def run_inference_task(
    raster_path: str,
    output_path: str,
    stats_path: str,
    serving_mode: str,
    checkpoint_path: str = "",
    triton_url: str = "",
    memmap: bool = False,
    tile: int = 256,
    overlap: int = 64,
    batch_size: int = 16,
    device: str = "cpu",
) -> str:
    """Run sliding-window inference over a raster with a local or Triton predictor."""
    if serving_mode == "triton":
        from libs.inference.predictors import TritonPredictor

        url = triton_url or TRITON_URL
        if not url:
            raise ValueError("Triton URL not set: pass triton_url or set TRITON_URL env var")
        predictor = TritonPredictor.connect(url)
    elif serving_mode == "local":
        from libs.inference.predictors import LocalTorchPredictor

        if not checkpoint_path:
            raise ValueError("checkpoint_path is required for serving_mode='local'")
        predictor = LocalTorchPredictor.from_checkpoint(checkpoint_path, device=device)
    else:
        raise ValueError(f"Unknown serving_mode: {serving_mode!r}")

    try:
        result = run_sliding_window(
            raster_path=Path(raster_path),
            output_path=Path(output_path),
            predictor=predictor,
            stats=load_stats(Path(stats_path)),
            tile=tile,
            overlap=overlap,
            batch_size=batch_size,
            memmap=memmap,
        )
    finally:
        close = getattr(predictor, "close", None)
        if callable(close):
            close()
    return str(result)


@task(name="postprocess-detections")
def postprocess_task(
    prob_raster_path: str,
    threshold: float = 0.5,
    min_area_px: int = 50,
    dedupe_method: str = "dbscan",
    dedupe_eps_m: float = 2000.0,
) -> list[dict]:
    """Extract, score, and deduplicate detections from probability raster."""
    detections = threshold_and_extract(prob_raster_path, threshold, min_area_px)
    detections = score_detections(detections)
    detections = deduplicate_detections(detections, method=dedupe_method, eps_m=dedupe_eps_m)

    gdf = detections_to_geodataframe(detections)
    return (
        json.loads(gdf.to_json())
        if not gdf.empty
        else {"type": "FeatureCollection", "features": []}
    )


@task(name="store-detections")
def store_detections_task(
    geojson: dict,
    model_version: str,
    db_url: str | None = None,
) -> int:
    """Store detections in PostGIS."""
    url = db_url or settings.database_url
    engine = create_engine(url)

    features = geojson.get("features", [])
    stored = 0

    with engine.begin() as conn:
        for feat in features:
            props = feat.get("properties", {})
            geom_json = json.dumps(feat["geometry"])
            conn.execute(
                text("""
                    INSERT INTO detections (geom, score, model_version, area_km2, compactness, evidence)
                    VALUES (
                        ST_SetSRID(ST_GeomFromGeoJSON(:geom), 4326),
                        :score, :model_version, :area_km2, :compactness, :evidence::jsonb
                    )
                """),
                {
                    "geom": geom_json,
                    "score": props.get("rank_score", props.get("score", 0)),
                    "model_version": model_version,
                    "area_km2": props.get("area_km2"),
                    "compactness": props.get("compactness"),
                    "evidence": json.dumps(props),
                },
            )
            stored += 1

    logger.info("Stored %d detections (model_version=%s)", stored, model_version)
    return stored
