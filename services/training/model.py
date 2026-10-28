"""UNet / DeepLabV3+ segmentation models for launch-site detection."""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch.nn as nn


def build_unet(
    in_channels: int = 10,
    classes: int = 1,
    encoder_name: str = "resnet34",
    encoder_weights: str | None = "imagenet",
) -> nn.Module:
    """Baseline UNet with a pretrained encoder.

    Args:
        in_channels: Number of input bands (6 S2 bands + 4 indices = 10).
        classes: Number of output classes (1 for binary segmentation).
        encoder_name: Backbone name from timm/smp.
        encoder_weights: Pretrained weights (None for random init).
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,
    )
    return model


def build_deeplabv3plus(
    in_channels: int = 10,
    classes: int = 1,
    encoder_name: str = "resnet50",
    encoder_weights: str | None = "imagenet",
) -> nn.Module:
    """Strong option: DeepLabV3+ for larger receptive field."""
    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,
    )
    return model
