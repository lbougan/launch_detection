"""Site-held-out train/val/test assignment for chips (decision 2).

Every chip of a held-out (val/test) AOI belongs to that AOI's split, however far it is from
the site. Other chips belong to the split of the nearest known site if that site is within
`exclusion_km`; otherwise they are background and go to train (background chips of training
AOIs train). So no training chip lies within `exclusion_km` of a validation or test site.
This holds when every held-out site is more than 2 x exclusion_km from every training site;
`tests/unit/test_manifests.py` enforces it.
"""

from __future__ import annotations

from libs.geo.projection import distance_m

SPLIT_KEYS = ("train", "val", "test")


def _centroid(bbox: list[float]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _site_split_map(sites: list[dict], splits: dict[str, list[str]]) -> dict[str, str]:
    bad_keys = [k for k, v in splits.items() if isinstance(v, list) and k not in SPLIT_KEYS]
    if bad_keys:
        raise ValueError(f"unknown split keys {bad_keys}; allowed: {list(SPLIT_KEYS)}")
    site_split = {name: split for split in SPLIT_KEYS for name in splits.get(split, [])}
    unknown = [s["name"] for s in sites if s["name"] not in site_split]
    if unknown:
        raise ValueError(f"sites missing from splits.json: {unknown}")
    return site_split


def _nearest_site(lon: float, lat: float, sites: list[dict]) -> tuple[dict, float]:
    nearest = min(sites, key=lambda s: distance_m(lon, lat, s["lon"], s["lat"]))
    return nearest, distance_m(lon, lat, nearest["lon"], nearest["lat"]) / 1000.0


def assign_splits(
    chips: list[dict],
    sites: list[dict],
    splits: dict[str, list[str]],
    exclusion_km: float = 20.0,
    aois: list[dict] | None = None,
) -> list[dict]:
    """Return new chip dicts with a 'split' key; inputs are not mutated.

    With `aois` (the AOI config entries, `name` and `bbox`), a chip whose `aoi` names a val/test
    AOI takes that split. Distances use each chip's `bbox_4326` (lon/lat, written by
    `chip_raster`), not `bbox`, which is in the raster CRS.
    """
    site_split = _site_split_map(sites, splits)
    held_out = {
        a["name"]: s for a in aois or [] if (s := aoi_split(a, sites, splits)) in ("val", "test")
    }
    out = []
    for chip in chips:
        if chip.get("aoi") in held_out:
            out.append({**chip, "split": held_out[chip["aoi"]]})
            continue
        if "bbox_4326" not in chip:
            raise ValueError("chip manifest predates bbox_4326; re-run tile_composite")
        lon, lat = _centroid(chip["bbox_4326"])
        nearest, d_km = _nearest_site(lon, lat, sites)
        split = site_split[nearest["name"]] if d_km <= exclusion_km else "train"
        out.append({**chip, "split": split})
    return out


def aoi_split(aoi: dict, sites: list[dict], splits: dict[str, list[str]]) -> str:
    """Split of the site nearest to the AOI bbox centre."""
    site_split = _site_split_map(sites, splits)
    lon, lat = _centroid(aoi["bbox"])
    nearest, _ = _nearest_site(lon, lat, sites)
    return site_split[nearest["name"]]


def count_by_split(chips: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in chips:
        counts[c["split"]] = counts.get(c["split"], 0) + 1
    return counts
