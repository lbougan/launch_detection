"""Chips + georeferenced label GeoTIFFs -> (image, mask, weight) tensors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from libs.features.indices import add_indices, normalize
from libs.features.stats import load_stats
from libs.geo.bandspec import read_bandspec
from libs.geo.labels import POSITIVE, read_label, weight_from_label


class LaunchSiteDataset(Dataset):
    """Each sample: image (C,H,W) standardised, mask (1,H,W) in {0,1}, weight (1,H,W) in {0,1}."""

    def __init__(
        self,
        chip_manifest: list[dict],
        stats_path: Path,
        split: str = "train",
        augment: bool = False,
    ):
        self.stats = load_stats(stats_path)
        self.augment = augment
        self.chips = [c for c in chip_manifest if c.get("split") == split]
        missing = [c["tile_id"] for c in self.chips if "label_path" not in c]
        if missing:
            raise ValueError(f"{len(missing)} chips have no label_path (run build_labels first)")

    def __len__(self) -> int:
        return len(self.chips)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        chip = self.chips[idx]
        with rasterio.open(chip["path"]) as src:
            image = src.read().astype(np.float32)
            spec = read_bandspec(src)
        image, _ = add_indices(image, spec.names)
        image = normalize(image, self.stats)
        image = np.nan_to_num(
            image, nan=0.0
        )  # cloud holes become the channel mean after standardisation

        label = read_label(chip["label_path"])
        weight = weight_from_label(label)
        mask = (label >= POSITIVE).astype(np.float32)

        if self.augment:
            image, mask, weight = _augment(image, mask, weight)

        return {
            "image": torch.from_numpy(image.astype(np.float32)),
            "mask": torch.from_numpy(mask[None]),
            "weight": torch.from_numpy(weight[None]),
            "meta": _meta(chip),
        }


def _meta(chip: dict) -> dict:
    meta = {"tile_id": chip["tile_id"], "bbox": chip["bbox"], "split": chip.get("split")}
    if "bbox_4326" in chip:
        meta["bbox_4326"] = chip["bbox_4326"]
    return meta


def _augment(image: np.ndarray, mask: np.ndarray, weight: np.ndarray):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(1, 2))
    mask, weight = np.rot90(mask, k), np.rot90(weight, k)
    if np.random.random() > 0.5:
        image, mask, weight = image[:, :, ::-1], mask[:, ::-1], weight[:, ::-1]
    if np.random.random() > 0.5:
        image, mask, weight = image[:, ::-1, :], mask[::-1, :], weight[::-1, :]
    return image.copy(), mask.copy(), weight.copy()


def collate(batch: list[dict]) -> dict:
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "weight": torch.stack([b["weight"] for b in batch]),
        "meta": [b["meta"] for b in batch],
    }
