"""Training entrypoint: build dataset, train model, log to MLflow."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader

from libs.config import settings
from services.training.dataset import LaunchSiteDataset
from services.training.lightning_module import LaunchSiteSegModule

logger = logging.getLogger(__name__)


@click.command()
@click.option("--manifest", type=click.Path(exists=True), required=True, help="Path to chip manifest JSON.")
@click.option("--sites", type=click.Path(exists=True), default="data/manifests/known_sites.json")
@click.option("--model-type", type=click.Choice(["unet", "deeplabv3+"]), default="unet")
@click.option("--encoder", default="resnet34")
@click.option("--in-channels", default=10, type=int)
@click.option("--lr", default=1e-3, type=float)
@click.option("--epochs", default=50, type=int)
@click.option("--batch-size", default=16, type=int)
@click.option("--gpus", default=0, type=int)
@click.option("--experiment-name", default="launchsite-seg")
def train(
    manifest: str,
    sites: str,
    model_type: str,
    encoder: str,
    in_channels: int,
    lr: float,
    epochs: int,
    batch_size: int,
    gpus: int,
    experiment_name: str,
) -> None:
    """Train the launch-site segmentation model."""
    with open(manifest) as f:
        chip_manifest = json.load(f)

    known_sites_path = Path(sites)

    train_ds = LaunchSiteDataset(chip_manifest, known_sites_path, split="train", augment=True)
    val_ds = LaunchSiteDataset(chip_manifest, known_sites_path, split="val", augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True,
        collate_fn=_collate,
    )

    module = LaunchSiteSegModule(
        model_type=model_type,
        encoder_name=encoder,
        in_channels=in_channels,
        lr=lr,
    )

    mlflow_logger = MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=settings.mlflow_tracking_uri,
    )

    callbacks = [
        ModelCheckpoint(
            monitor="val/loss",
            mode="min",
            save_top_k=3,
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
    logger.info("Training complete. Best checkpoint: %s", trainer.checkpoint_callback.best_model_path)


def _collate(batch: list[dict]) -> dict:
    """Custom collate that handles the meta dict."""
    import torch
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "meta": [b["meta"] for b in batch],
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
