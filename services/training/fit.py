"""The one training loop (A13). Both the CLI and the Prefect task call `fit`."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FitConfig:
    manifest_path: str
    stats_path: str
    known_sites_path: str = "data/manifests/known_sites.json"
    splits_path: str = "data/manifests/splits.json"
    model_type: str = "unet"
    encoder: str = "resnet34"
    encoder_weights: str | None = "imagenet"
    lr: float = 1e-3
    epochs: int = 50
    batch_size: int = 16
    gpus: int = 0
    num_workers: int = 2
    precision: int | str = 32
    monitor: str = "val/site_recall@10"
    experiment_name: str = "launchsite-seg"
    checkpoint_dir: str = "checkpoints"
    mlflow_tracking_uri: str | None = None  # None -> no MLflow logger (tests, CI)
    limit_train_batches: float | int = 1.0
    limit_val_batches: float | int = 1.0
    val_composites: tuple[str, ...] = ()  # composites of validation AOIs for site-level metrics


def resolve_monitor(cfg: FitConfig) -> tuple[str, str]:
    """(monitor, mode); a recall monitor without val composites falls back to val/loss."""
    if "site_recall" in cfg.monitor and not cfg.val_composites:
        logger.warning(
            "monitor=%s needs val_composites; falling back to val/loss (mode=min)", cfg.monitor
        )
        return "val/loss", "min"
    return cfg.monitor, "max" if "recall" in cfg.monitor else "min"


def checkpoint_filename(monitor: str) -> str:
    """Template using the exact logged key (use with auto_insert_metric_name=False)."""
    return "launchsite-{epoch:02d}-{" + monitor + ":.3f}"


def fit(cfg: FitConfig) -> str:
    """Train and return the best checkpoint path."""
    return fit_with_trainer(cfg)[0]


def fit_with_trainer(cfg: FitConfig):
    """Train; return (best checkpoint path, the pl.Trainer) so callers can inspect metrics."""
    import pytorch_lightning as pl
    import torch
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from torch.utils.data import DataLoader

    from services.training.dataset import LaunchSiteDataset, collate
    from services.training.lightning_module import LaunchSiteSegModule

    chips = json.loads(Path(cfg.manifest_path).read_text())
    train_ds = LaunchSiteDataset(chips, Path(cfg.stats_path), split="train", augment=True)
    val_ds = LaunchSiteDataset(chips, Path(cfg.stats_path), split="val", augment=False)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(f"empty split: train={len(train_ds)} val={len(val_ds)}")
    pin = cfg.gpus > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        collate_fn=collate,
    )

    module = LaunchSiteSegModule(
        model_type=cfg.model_type,
        encoder_name=cfg.encoder,
        encoder_weights=cfg.encoder_weights,
        in_channels=10,
        lr=cfg.lr,
        stats_path=cfg.stats_path,
        known_sites_path=cfg.known_sites_path,
        splits_path=cfg.splits_path,
        val_composites=list(cfg.val_composites),
    )
    logger_ = None
    if cfg.mlflow_tracking_uri:
        from pytorch_lightning.loggers import MLFlowLogger

        logger_ = MLFlowLogger(
            experiment_name=cfg.experiment_name, tracking_uri=cfg.mlflow_tracking_uri
        )
    monitor, mode = resolve_monitor(cfg)
    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.checkpoint_dir,
            monitor=monitor,
            mode=mode,
            save_top_k=3,
            filename=checkpoint_filename(monitor),
            auto_insert_metric_name=False,
        ),
        EarlyStopping(monitor=monitor, patience=10, mode=mode),
    ]
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator="gpu" if cfg.gpus > 0 else "cpu",
        devices=cfg.gpus if cfg.gpus > 0 else 1,
        logger=logger_ if logger_ is not None else False,
        callbacks=callbacks,
        log_every_n_steps=10,
        precision=cfg.precision,
        limit_train_batches=cfg.limit_train_batches,
        limit_val_batches=cfg.limit_val_batches,
        enable_progress_bar=False,
        deterministic=True,
    )
    torch.manual_seed(0)
    trainer.fit(module, train_loader, val_loader)
    best = trainer.checkpoint_callback.best_model_path
    logger.info("best checkpoint: %s", best)
    return best, trainer
