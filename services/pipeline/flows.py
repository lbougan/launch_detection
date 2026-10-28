"""Prefect flows: orchestrate the full pipeline."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from prefect import flow

from libs.config import default_stats_path, settings
from libs.geo.chips import load_known_sites
from libs.geo.splits import aoi_split, count_by_split
from services.pipeline.tasks import (
    assign_splits_task,
    build_composite_task,
    build_labels_task,
    compute_stats_task,
    ingest_aoi_task,
    postprocess_task,
    run_inference_task,
    store_detections_task,
    tile_composite_task,
    train_model_task,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

logger = logging.getLogger(__name__)


@flow(name="ingest-and-preprocess", log_prints=True)
def ingest_and_preprocess_flow(
    aoi_config_path: str = "data/manifests/aoi_config.json",
    date_range: str = "2024-01-01/2024-06-30",
    known_sites_path: str = "data/manifests/known_sites.json",
    splits_path: str = "data/manifests/splits.json",
    max_cloud_cover: float = 20.0,
) -> str:
    """Flow 1: Ingest imagery, build composites, tile, and create labels.

    Returns the path to the chip manifest JSON.
    """
    with open(aoi_config_path) as f:
        aoi_config = json.load(f)

    data_dir = Path(settings.data_dir)
    all_chips = []
    training_composites: list[str] = []
    sites = load_known_sites(Path(known_sites_path))
    with open(splits_path) as f:
        splits = json.load(f)

    for aoi in aoi_config["aois"]:
        aoi_name = aoi["name"]
        bbox = aoi["bbox"]
        aoi_dir = data_dir / "raw" / aoi_name

        scene_paths = ingest_aoi_task(
            aoi_name=aoi_name,
            bbox=bbox,
            date_range=date_range,
            output_dir=str(aoi_dir),
            max_cloud_cover=max_cloud_cover,
        )

        s2_paths = [p for p in scene_paths if "sentinel2" in p]
        if not s2_paths:
            logger.warning("No S2 scenes for AOI %s, skipping", aoi_name)
            continue

        composite_path = build_composite_task(
            scene_paths=s2_paths,
            output_path=str(data_dir / "composites" / f"{aoi_name}_composite.tif"),
        )

        if aoi_split(aoi, sites, splits) == "train":
            training_composites.append(composite_path)

        chips = tile_composite_task(
            composite_path=composite_path,
            output_dir=str(data_dir / "tiles" / aoi_name),
        )

        for chip in chips:
            chip["aoi"] = aoi_name
        all_chips.extend(chips)

    split_chips = assign_splits_task(
        chip_manifest=all_chips,
        known_sites_path=known_sites_path,
        splits_path=splits_path,
    )
    logger.info("Chips per split: %s", count_by_split(split_chips))

    labeled_chips = build_labels_task(
        chip_manifest=split_chips,
        known_sites_path=known_sites_path,
    )

    stats_path = str(default_stats_path())
    if training_composites:
        compute_stats_task(
            composite_paths=training_composites,
            output_path=stats_path,
        )
    else:
        logger.warning("No training composites; channel stats not computed")

    manifest_path = str(data_dir / "manifests" / "chip_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(labeled_chips, f, indent=2)
    logger.info("stats_path=%s splits_path=%s", stats_path, splits_path)

    print(f"Created chip manifest with {len(labeled_chips)} tiles → {manifest_path}")
    return manifest_path


@flow(name="train-model", log_prints=True)
def train_flow(
    manifest_path: str = "data/manifests/chip_manifest.json",
    stats_path: str = str(default_stats_path()),
    model_type: str = "unet",
    encoder: str = "resnet34",
    lr: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 16,
    gpus: int = 0,
    experiment_name: str = "launchsite-seg",
) -> str:
    """Flow 2: Train the segmentation model.

    Returns the path to the best checkpoint.
    """
    best_ckpt = train_model_task(
        manifest_path=manifest_path,
        stats_path=stats_path,
        model_type=model_type,
        encoder=encoder,
        lr=lr,
        epochs=epochs,
        batch_size=batch_size,
        gpus=gpus,
        experiment_name=experiment_name,
    )
    print(f"Training complete. Best checkpoint: {best_ckpt}")
    return best_ckpt


@flow(name="scan-and-detect", log_prints=True)
def scan_and_detect_flow(
    aoi_config_path: str = "data/manifests/aoi_config.json",
    checkpoint_path: str = "checkpoints/best.ckpt",
    model_version: str = "v0.1",
    threshold: float = 0.5,
    stats_path: str = str(default_stats_path()),
    device: str = "cpu",
    serving_mode: str = "",
    triton_url: str = "",
) -> int:
    """Run inference over AOI composites, postprocess, and store detections.

    Args:
        serving_mode: ``"local"`` for in-process PyTorch, ``"triton"`` for
            Triton Inference Server. Defaults to ``"triton"`` when the
            ``TRITON_URL`` env var is set, otherwise ``"local"``.
        device: Torch device for ``serving_mode='local'``.
        stats_path: Channel stats written at training time (must match the checkpoint).
        triton_url: Triton gRPC endpoint (overrides ``TRITON_URL`` env var).

    Returns:
        Total number of stored detections.
    """
    if not serving_mode:
        serving_mode = "triton" if os.environ.get("TRITON_URL") else "local"

    with open(aoi_config_path) as f:
        aoi_config = json.load(f)

    data_dir = Path(settings.data_dir)
    total_stored = 0

    for aoi in aoi_config["aois"]:
        aoi_name = aoi["name"]
        composite_path = data_dir / "composites" / f"{aoi_name}_composite.tif"

        if not composite_path.exists():
            logger.warning("No composite for AOI %s, skipping inference", aoi_name)
            continue

        prob_path = str(data_dir / "predictions" / f"{aoi_name}_prob.tif")

        run_inference_task(
            raster_path=str(composite_path),
            output_path=prob_path,
            stats_path=stats_path,
            serving_mode=serving_mode,
            checkpoint_path=checkpoint_path,
            triton_url=triton_url,
            device=device,
        )

        geojson = postprocess_task(
            prob_raster_path=prob_path,
            threshold=threshold,
        )

        stored = store_detections_task(
            geojson=geojson,
            model_version=model_version,
        )
        total_stored += stored

    print(f"Stored {total_stored} total detections across all AOIs")
    return total_stored


@flow(name="full-pipeline", log_prints=True)
def full_pipeline_flow(
    aoi_config_path: str = "data/manifests/aoi_config.json",
    date_range: str = "2024-01-01/2024-06-30",
    known_sites_path: str = "data/manifests/known_sites.json",
    checkpoint_path: str = "checkpoints/best.ckpt",
    model_version: str = "v0.1",
    skip_ingest: bool = False,
    skip_training: bool = False,
    epochs: int = 30,
    batch_size: int = 16,
    gpus: int = 0,
) -> None:
    """End-to-end pipeline: ingest → preprocess → train → infer → store."""
    manifest_path = "data/manifests/chip_manifest.json"

    if not skip_ingest:
        manifest_path = ingest_and_preprocess_flow(
            aoi_config_path=aoi_config_path,
            date_range=date_range,
            known_sites_path=known_sites_path,
        )
        print(f"Ingest+preprocess done: {manifest_path}")

    if not skip_training:
        best_ckpt = train_flow(
            manifest_path=manifest_path,
            stats_path=str(default_stats_path()),
            epochs=epochs,
            batch_size=batch_size,
            gpus=gpus,
        )
        checkpoint_path = best_ckpt
        print(f"Training done. Using checkpoint: {checkpoint_path}")

    total = scan_and_detect_flow(
        aoi_config_path=aoi_config_path,
        checkpoint_path=checkpoint_path,
        model_version=model_version,
    )
    print(f"Pipeline complete. {total} detections stored.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch-site detection Prefect pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_full = sub.add_parser("full", help="Run the full end-to-end pipeline")
    p_full.add_argument("--aoi-config", default="data/manifests/aoi_config.json")
    p_full.add_argument("--date-range", default="2024-01-01/2024-06-30")
    p_full.add_argument("--known-sites", default="data/manifests/known_sites.json")
    p_full.add_argument("--checkpoint", default="checkpoints/best.ckpt")
    p_full.add_argument("--model-version", default="v0.1")
    p_full.add_argument("--skip-ingest", action="store_true")
    p_full.add_argument("--skip-training", action="store_true")
    p_full.add_argument("--epochs", type=int, default=30)
    p_full.add_argument("--batch-size", type=int, default=16)
    p_full.add_argument("--gpus", type=int, default=0)

    p_ingest = sub.add_parser("ingest", help="Run ingest + preprocess only")
    p_ingest.add_argument("--aoi-config", default="data/manifests/aoi_config.json")
    p_ingest.add_argument("--date-range", default="2024-01-01/2024-06-30")
    p_ingest.add_argument("--known-sites", default="data/manifests/known_sites.json")

    p_train = sub.add_parser("train", help="Run training only")
    p_train.add_argument("--manifest", default="data/manifests/chip_manifest.json")
    p_train.add_argument("--stats", default=str(default_stats_path()))
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--batch-size", type=int, default=16)
    p_train.add_argument("--gpus", type=int, default=0)

    p_detect = sub.add_parser("detect", help="Run inference + detection only")
    p_detect.add_argument("--aoi-config", default="data/manifests/aoi_config.json")
    p_detect.add_argument("--checkpoint", default="checkpoints/best.ckpt")
    p_detect.add_argument("--model-version", default="v0.1")
    p_detect.add_argument(
        "--serving-mode",
        default="",
        choices=["", "local", "triton"],
        help="Inference backend: 'local' (in-process PyTorch) or 'triton' (Triton server). "
        "Default: auto-detect via TRITON_URL env var.",
    )
    p_detect.add_argument("--stats", default=str(default_stats_path()))
    p_detect.add_argument("--device", default="cpu", help="Torch device for local serving")
    p_detect.add_argument("--triton-url", default="", help="Triton gRPC endpoint")

    args = parser.parse_args()

    if args.command == "full":
        full_pipeline_flow(
            aoi_config_path=args.aoi_config,
            date_range=args.date_range,
            known_sites_path=args.known_sites,
            checkpoint_path=args.checkpoint,
            model_version=args.model_version,
            skip_ingest=args.skip_ingest,
            skip_training=args.skip_training,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gpus=args.gpus,
        )
    elif args.command == "ingest":
        ingest_and_preprocess_flow(
            aoi_config_path=args.aoi_config,
            date_range=args.date_range,
            known_sites_path=args.known_sites,
        )
    elif args.command == "train":
        train_flow(
            manifest_path=args.manifest,
            stats_path=args.stats,
            epochs=args.epochs,
            batch_size=args.batch_size,
            gpus=args.gpus,
        )
    elif args.command == "detect":
        scan_and_detect_flow(
            aoi_config_path=args.aoi_config,
            checkpoint_path=args.checkpoint,
            model_version=args.model_version,
            stats_path=args.stats,
            device=args.device,
            serving_mode=args.serving_mode,
            triton_url=args.triton_url,
        )
