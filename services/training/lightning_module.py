"""PyTorch Lightning module for launch-site segmentation."""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics import AUROC, F1Score, Precision, Recall

from services.training.losses import CombinedLoss
from services.training.model import build_deeplabv3plus, build_unet


class LaunchSiteSegModule(pl.LightningModule):
    """Lightning wrapper around the segmentation model.

    Handles training, validation, metrics, and optimizer config.
    """

    def __init__(
        self,
        model_type: str = "unet",
        encoder_name: str = "resnet34",
        in_channels: int = 10,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        dice_weight: float = 0.5,
        label_smoothing: float = 0.05,
        scheduler_patience: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters()

        if model_type == "unet":
            self.model = build_unet(in_channels, 1, encoder_name)
        elif model_type == "deeplabv3+":
            self.model = build_deeplabv3plus(in_channels, 1, encoder_name)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.loss_fn = CombinedLoss(focal_alpha, focal_gamma, dice_weight, label_smoothing)

        self.val_f1 = F1Score(task="binary")
        self.val_precision = Precision(task="binary")
        self.val_recall = Recall(task="binary")
        self.val_auroc = AUROC(task="binary")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        logits = self(batch["image"])
        loss = self.loss_fn(logits, batch["mask"])
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        logits = self(batch["image"])
        loss = self.loss_fn(logits, batch["mask"])

        preds = (torch.sigmoid(logits) > 0.5).long().flatten()
        targets = (batch["mask"] > 0.5).long().flatten()

        self.val_f1.update(preds, targets)
        self.val_precision.update(preds, targets)
        self.val_recall.update(preds, targets)

        probs_flat = torch.sigmoid(logits).flatten()
        if targets.sum() > 0 and (targets == 0).sum() > 0:
            self.val_auroc.update(probs_flat, targets)

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        self.log("val/f1", self.val_f1.compute(), prog_bar=True)
        self.log("val/precision", self.val_precision.compute())
        self.log("val/recall", self.val_recall.compute())
        try:
            self.log("val/auroc", self.val_auroc.compute())
        except Exception:
            pass

        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_auroc.reset()

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=self.hparams.scheduler_patience,
            factor=0.5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"},
        }
