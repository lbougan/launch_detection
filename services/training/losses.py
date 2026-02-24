"""Loss functions for weak-label segmentation training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss for handling class imbalance.

    Focal loss down-weights easy examples so the model focuses on
    hard positives (actual launch-site pixels that are rare).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, label_smoothing: float = 0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, H, W) raw model output.
            targets: (B, 1, H, W) float in [0, 1] (supports soft/smoothed labels).
        """
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * focal_weight * bce

        return loss.mean()


class CombinedLoss(nn.Module):
    """Focal loss + Dice loss for balanced segmentation training."""

    def __init__(
        self,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        dice_weight: float = 0.5,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.focal = FocalLoss(focal_alpha, focal_gamma, label_smoothing)
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        focal_loss = self.focal(logits, targets)
        dice_loss = self._soft_dice(logits, targets)
        return focal_loss + self.dice_weight * dice_loss

    @staticmethod
    def _soft_dice(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        return 1 - (2 * intersection + eps) / (union + eps)
