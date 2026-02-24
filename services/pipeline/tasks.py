"""Prefect tasks: atomic units of work in the pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import rasterio
import torch
from prefect import task
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from sqlalchemy import create_engine, text
from torch.utils.data import DataLoader

from libs.config import settings
from libs.features.indices import add_indices, build_temporal_composite, normalize_percentile
from libs.geo.masks import apply_cloud_mask
from libs.geo.postprocess import (
    deduplicate_detections,
    detections_to_geodataframe,
    score_detections,
    threshold_and_extract,
)
from libs.geo.tiling import build_dataset_split, build_weak_label_mask, chip_raster, load_known_sites
from libs.stac.client import ingest_aoi
from services.training.dataset import LaunchSiteDataset
from services.training.inference import load_model, sliding_window_inference
from services.training.lightning_module import LaunchSiteSegModule

logger = logging.getLogger(__name__)


@task(name="train-model")
def train_model_task(
    manifest_path: str,
    known_sites_path: str = "data/manifests/known_sites.json",
    model_type: str = "unet",
    encoder: str = "resnet34",
    in_channels: int = 10,
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 16,
    gpus: int = 0,
    experiment_name: str = "launchsite-seg",
) -> str:
    """Train the segmentation model and return the best checkpoint path."""
    with open(manifest_path) as f:
        chip_manifest = json.load(f)

    sites = Path(known_sites_path)

    train_ds = LaunchSiteDataset(chip_manifest, sites, split="train", augment=True)
    val_ds = LaunchSiteDataset(chip_manifest, sites, split="val", augment=False)

    def _collate(batch: list[dict]) -> dict:
        return {
            "image": torch.stack([b["image"] for b in batch]),
            "mask": torch.stack([b["mask"] for b in batch]),
            "meta": [b["meta"] for b in batch],
        }

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=_collate,
    )

    module = LaunchSiteSegModule(
        model_type=model_type, encoder_name=encoder,
        in_channels=in_channels, lr=lr,
    )

    mlflow_logger = MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=settings.mlflow_tracking_uri,
    )

    callbacks = [
        ModelCheckpoint(
            monitor="val/loss", mode="min", save_top_k=3,
            filename="launchsite-{epoch:02d}-{val/loss:.4f}",
        ),
        EarlyStopping(monitor="val/loss", patience=10, mode="min"),
    ]

    accelerator = "gpu" if gpus > 0 else "cpu"
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=gpus if gpus > 0 else "auto",
        logger=mlflow_logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        precision="16-mixed" if gpus > 0 else 32,
    )

    logger.info("Starting training: model=%s, encoder=%s, epochs=%d", model_type, encoder, epochs)
    trainer.fit(module, train_loader, val_loader)

    best = trainer.checkpoint_callback.best_model_path
    logger.info("Training complete. Best checkpoint: %s", best)
    return best


@task(name="ingest-aoi", retries=2, retry_delay_seconds=60)
def ingest_aoi_task(
    aoi_name: str,
    bbox: list[float],
    date_range: str,
    output_dir: str,
    max_cloud_cover: float = 20.0,
    include_s1: bool = False,
) -> list[str]:
    """Ingest satellite imagery for a single AOI."""
    paths = ingest_aoi(
        bbox=bbox,
        date_range=date_range,
        output_dir=Path(output_dir),
        max_cloud_cover=max_cloud_cover,
        include_s1=include_s1,
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

    composite, profile = build_temporal_composite(paths, method)
    composite = normalize_percentile(composite)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    profile.update(dtype="float32", compress="deflate")
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(composite)

    logger.info("Built composite → %s", out)
    return str(out)


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


@task(name="build-labels")
def build_labels_task(
    chip_manifest: list[dict],
    known_sites_path: str,
) -> list[dict]:
    """Generate weak label masks and dataset splits."""
    sites = load_known_sites(Path(known_sites_path))
    enriched = build_dataset_split(chip_manifest, sites)

    for chip in enriched:
        mask = build_weak_label_mask(
            bbox=chip["bbox"],
            tile_size=256,
            known_sites=sites,
        )
        label_path = Path(chip["path"]).parent / f"{chip['tile_id']}_label.npy"
        np.save(label_path, mask)
        chip["label_path"] = str(label_path)

    return enriched


@task(name="run-inference")
def run_inference_task(
    checkpoint_path: str,
    raster_path: str,
    output_path: str,
    tile_size: int = 256,
    overlap: int = 64,
    device: str = "cpu",
) -> str:
    """Run sliding-window inference over a raster."""
    model = load_model(checkpoint_path, device)
    result = sliding_window_inference(
        model=model,
        raster_path=Path(raster_path),
        output_path=Path(output_path),
        tile_size=tile_size,
        overlap=overlap,
        device=device,
    )
    return str(result)


@task(name="postprocess-detections")
def postprocess_task(
    prob_raster_path: str,
    threshold: float = 0.5,
    min_area_px: int = 50,
    dedupe_method: str = "dbscan",
    dedupe_eps: float = 0.02,
) -> list[dict]:
    """Extract, score, and deduplicate detections from probability raster."""
    detections = threshold_and_extract(prob_raster_path, threshold, min_area_px)
    detections = score_detections(detections)
    detections = deduplicate_detections(detections, method=dedupe_method, eps_deg=dedupe_eps)

    gdf = detections_to_geodataframe(detections)
    return json.loads(gdf.to_json()) if not gdf.empty else {"type": "FeatureCollection", "features": []}


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
                    "score": props.get("score", 0),
                    "model_version": model_version,
                    "area_km2": props.get("area_km2"),
                    "compactness": props.get("compactness"),
                    "evidence": json.dumps(props),
                },
            )
            stored += 1

    logger.info("Stored %d detections (model_version=%s)", stored, model_version)
    return stored
