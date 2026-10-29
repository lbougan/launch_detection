"""Site-level evaluation (model card): recall@K, FP per 1000 km², reliability on `prob`."""

from __future__ import annotations

import numpy as np

from libs.geo.projection import distance_m


def _matches(pred: dict, sites: list[dict], match_m: float) -> bool:
    return any(
        distance_m(pred["centroid_lon"], pred["centroid_lat"], s["lon"], s["lat"]) <= match_m
        for s in sites
    )


def _ranked(preds: list[dict]) -> list[dict]:
    return sorted(
        preds,
        key=lambda p: p.get("rank_score") if p.get("rank_score") is not None else p["prob"],
        reverse=True,
    )


def site_recall_at_k(
    preds: list[dict], sites: list[dict], k: int = 10, match_m: float = 5000.0
) -> float:
    if not sites:
        return 0.0
    top = _ranked(preds)[:k]
    hit = sum(
        1
        for s in sites
        if any(
            distance_m(p["centroid_lon"], p["centroid_lat"], s["lon"], s["lat"]) <= match_m
            for p in top
        )
    )
    return hit / len(sites)


def fp_per_1000km2(
    preds: list[dict], sites: list[dict], scanned_km2: float, match_m: float = 5000.0
) -> float:
    if scanned_km2 <= 0:
        return 0.0
    fp = sum(1 for p in preds if not _matches(p, sites, match_m))
    return fp / (scanned_km2 / 1000.0)


def reliability_bins(
    preds: list[dict], sites: list[dict], n_bins: int = 10, match_m: float = 5000.0
) -> dict:
    """Calibration of `prob` (never rank_score): per confidence bin, fraction that match a site."""
    probs = np.array([p["prob"] for p in preds], dtype=float)
    is_tp = np.array([_matches(p, sites, match_m) for p in preds], dtype=bool)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    acc, conf, count = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = (probs >= lo) & ((probs < hi) if hi < 1.0 else (probs <= hi))
        count.append(int(sel.sum()))
        acc.append(float(is_tp[sel].mean()) if sel.any() else 0.0)
        conf.append(float(probs[sel].mean()) if sel.any() else (lo + hi) / 2)
    return {
        "bin_edges": edges.tolist(),
        "bin_accuracy": acc,
        "bin_confidence": conf,
        "bin_count": count,
    }
