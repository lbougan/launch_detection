"""UNet / DeepLabV3+ segmentation models for launch-site detection."""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
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


class DualEncoderUNet(nn.Module):
    """Two-branch encoder UNet for Sentinel-2 + Sentinel-1 fusion.

    Branch A: optical bands (S2)
    Branch B: SAR bands (S1 VV + VH)
    Features are concatenated at the bottleneck and decoded jointly.
    """

    def __init__(
        self,
        optical_channels: int = 10,
        sar_channels: int = 2,
        classes: int = 1,
        encoder_name: str = "resnet34",
    ):
        super().__init__()
        self.optical_encoder = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=optical_channels,
            classes=classes,
            activation=None,
        )
        self.sar_encoder = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=sar_channels,
            classes=classes,
            activation=None,
        )
        bottleneck_ch = self._get_bottleneck_channels(encoder_name)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(bottleneck_ch * 2, bottleneck_ch, 1),
            nn.BatchNorm2d(bottleneck_ch),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _get_bottleneck_channels(encoder_name: str) -> int:
        defaults = {"resnet18": 512, "resnet34": 512, "resnet50": 2048}
        return defaults.get(encoder_name, 512)

    def forward(self, optical: torch.Tensor, sar: torch.Tensor) -> torch.Tensor:
        opt_features = self.optical_encoder.encoder(optical)
        sar_features = self.sar_encoder.encoder(sar)

        fused_bottleneck = self.fusion_conv(
            torch.cat([opt_features[-1], sar_features[-1]], dim=1)
        )
        opt_features[-1] = fused_bottleneck

        decoder_output = self.optical_encoder.decoder(*opt_features)
        logits = self.optical_encoder.segmentation_head(decoder_output)
        return logits
