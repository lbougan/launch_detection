"""Site-held-out train/val/test assignment for chips (decision 2).

A chip belongs to the split of the nearest known site if that site is within
`exclusion_km`; otherwise it is background and goes to train. So no training chip lies
within `exclusion_km` of a validation or test site.
"""

from __future__ import annotations

from libs.geo.projection import distance_m


def _centroid(bbox: list[float]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _site_split_map(sites: list[dict], splits: dict[str, list[str]]) -> dict[str, str]:
    site_split = {
        name: split for split, names in splits.items() if isinstance(names, list) for name in names
    }
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
) -> list[dict]:
    """Return new chip dicts with a 'split' key; inputs are not mutated."""
    site_split = _site_split_map(sites, splits)
    out = []
    for chip in chips:
        lon, lat = _centroid(chip["bbox"])
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
