"""PyTorch Lightning module for launch-site segmentation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from torchmetrics import AUROC, F1Score, Precision, Recall

from libs.config import default_stats_path
from services.training.losses import CombinedLoss
from services.training.model import build_deeplabv3plus, build_unet

logger = logging.getLogger(__name__)


class LaunchSiteSegModule(pl.LightningModule):
    """Lightning wrapper around the segmentation model.

    Handles training, validation, metrics, and optimizer config.
    """

    def __init__(
        self,
        model_type: str = "unet",
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        in_channels: int = 10,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        dice_weight: float = 0.5,
        scheduler_patience: int = 5,
        stats_path: str = str(default_stats_path()),
        stats: dict | None = None,
        known_sites_path: str = "data/manifests/known_sites.json",
        splits_path: str = "data/manifests/splits.json",
        val_composites: list[str] | None = None,
    ):
        super().__init__()
        val_composites = list(val_composites or [])
        self.save_hyperparameters()

        if model_type == "unet":
            self.model = build_unet(in_channels, 1, encoder_name, encoder_weights)
        elif model_type == "deeplabv3+":
            self.model = build_deeplabv3plus(in_channels, 1, encoder_name, encoder_weights)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.loss_fn = CombinedLoss(focal_alpha, focal_gamma, dice_weight)

        self.val_f1 = F1Score(task="binary")
        self.val_precision = Precision(task="binary")
        self.val_recall = Recall(task="binary")
        self.val_auroc = AUROC(task="binary")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        logits = self(batch["image"])
        loss = self.loss_fn(logits, batch["mask"], batch["weight"])
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        logits = self(batch["image"])
        loss = self.loss_fn(logits, batch["mask"], batch["weight"])

        valid = (batch["weight"] > 0).flatten()
        preds = (torch.sigmoid(logits) > 0.5).long().flatten()[valid]
        targets = (batch["mask"] >= 0.5).long().flatten()[valid]

        self.val_f1.update(preds, targets)
        self.val_precision.update(preds, targets)
        self.val_recall.update(preds, targets)

        probs_flat = torch.sigmoid(logits).flatten()[valid]
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
        self._log_site_metrics()

    def _log_site_metrics(self) -> None:
        """Log val/site_recall@10 every epoch (0.0 without val composites) for the checkpoint monitor.

        Site-level evaluation runs on global rank 0 only; the result is broadcast so every rank
        logs the same value (``sync_dist=False``) and the checkpoint monitor exists everywhere.
        Failures propagate; a broken evaluation must not silently log 0.0 into the monitor.
        """
        if self.trainer.sanity_checking:
            return
        metrics = (0.0, 0.0)
        if self.hparams.val_composites and self.trainer.is_global_zero:
            metrics = self._site_metrics()
        recall, fp = self.trainer.strategy.broadcast(metrics, src=0)
        self.log("val/site_recall@10", recall, prog_bar=True, sync_dist=False)
        self.log("val/fp_per_1000km2", fp, sync_dist=False)

    def _site_metrics(self) -> tuple[float, float]:
        import json
        import tempfile

        from libs.geo.chips import load_known_sites
        from libs.inference.predictors import LocalTorchPredictor
        from services.training.evaluate import evaluate_composites

        comps = [Path(c) for c in self.hparams.val_composites]
        missing = [c for c in comps if not c.exists()]
        if missing:
            raise FileNotFoundError(f"val composites missing: {missing}")
        splits = json.loads(Path(self.hparams.splits_path).read_text())
        names = set(splits["val"])
        sites = [
            s for s in load_known_sites(Path(self.hparams.known_sites_path)) if s["name"] in names
        ]
        stats = self.hparams.get("stats") or Path(self.hparams.stats_path)
        predictor = LocalTorchPredictor(self, device=str(self.device))
        with tempfile.TemporaryDirectory() as tmp:
            res = evaluate_composites(predictor, comps, stats, sites, Path(tmp), "val")
        return res.recall_at_10, res.fp_per_1000km2

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
