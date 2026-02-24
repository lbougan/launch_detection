"""PyTorch dataset for launch-site segmentation tiles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from libs.features.indices import add_indices, normalize_percentile
from libs.geo.tiling import build_weak_label_mask, load_known_sites


class LaunchSiteDataset(Dataset):
    """Dataset that loads tiled imagery chips and weak label masks.

    Each sample returns:
        image: (C, H, W) float32 tensor — normalized bands + indices
        mask:  (1, H, W) float32 tensor — weak label (0/1/0.5)
        meta:  dict with tile_id, bbox, split
    """

    def __init__(
        self,
        chip_manifest: list[dict],
        known_sites_path: Path,
        split: str = "train",
        use_indices: bool = True,
        augment: bool = False,
    ):
        self.known_sites = load_known_sites(known_sites_path)
        self.use_indices = use_indices
        self.augment = augment
        self.chips = [c for c in chip_manifest if c.get("split", "train") == split]

    def __len__(self) -> int:
        return len(self.chips)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        chip = self.chips[idx]

        with rasterio.open(chip["path"]) as src:
            image = src.read().astype(np.float32)

        image = normalize_percentile(image)

        if self.use_indices and image.shape[0] == 6:
            image = add_indices(image)

        mask = build_weak_label_mask(
            bbox=chip["bbox"],
            tile_size=image.shape[-1],
            known_sites=self.known_sites,
        )

        if self.augment:
            image, mask = self._augment(image, mask)

        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask[None]),  # (1, H, W)
            "meta": {
                "tile_id": chip["tile_id"],
                "bbox": chip["bbox"],
                "split": chip.get("split", "train"),
            },
        }

    @staticmethod
    def _augment(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Simple geometric augmentations (random flips and 90-degree rotations)."""
        k = np.random.randint(0, 4)
        image = np.rot90(image, k, axes=(1, 2)).copy()
        mask = np.rot90(mask, k, axes=(0, 1)).copy()

        if np.random.random() > 0.5:
            image = np.flip(image, axis=2).copy()
            mask = np.flip(mask, axis=1).copy()

        if np.random.random() > 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=0).copy()

        return image, mask
