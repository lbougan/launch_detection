"""Focal + soft Dice loss with a per-pixel weight (0 on the ignore ring)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    def __init__(
        self, focal_alpha: float = 0.25, focal_gamma: float = 2.0, dice_weight: float = 0.5
    ):
        super().__init__()
        self.alpha = focal_alpha
        self.gamma = focal_gamma
        self.dice_weight = dice_weight

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        """logits/targets/weight: (B,1,H,W); targets in {0,1}; weight in {0,1}."""
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal = alpha_t * (1 - p_t) ** self.gamma * bce
        focal = (focal * weight).sum() / weight.sum().clamp_min(1.0)

        eps = 1e-6
        pw, tw = probs * weight, targets * weight
        intersection = (pw * tw).sum()
        dice = 1 - (2 * intersection + eps) / (pw.sum() + tw.sum() + eps)
        return focal + self.dice_weight * dice
