"""Custom evaluation metrics for launch-site detection.

Implements the spec's key metrics:
  - Site-level recall@K
  - False positive rate per 1000 km^2
  - Calibration (confidence vs precision)
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Point


def site_level_recall_at_k(
    predicted_polygons: list[dict],
    known_sites: list[dict],
    k: int = 10,
    match_distance_km: float = 5.0,
) -> float:
    """What fraction of known sites appear in the top-K ranked predictions?

    Args:
        predicted_polygons: Sorted by score (descending). Each has 'centroid_lon', 'centroid_lat'.
        known_sites: Each has 'lon', 'lat'.
        k: Number of top predictions to consider.
        match_distance_km: Max distance for a prediction to "match" a site.

    Returns:
        Recall value in [0, 1].
    """
    if not known_sites:
        return 0.0

    match_deg = match_distance_km / 111.0
    top_k = predicted_polygons[:k]
    matched = 0

    for site in known_sites:
        site_pt = Point(site["lon"], site["lat"])
        for pred in top_k:
            pred_pt = Point(pred["centroid_lon"], pred["centroid_lat"])
            if site_pt.distance(pred_pt) < match_deg:
                matched += 1
                break

    return matched / len(known_sites)


def false_positive_rate_per_1000km2(
    predicted_polygons: list[dict],
    known_sites: list[dict],
    total_area_km2: float,
    match_distance_km: float = 5.0,
) -> float:
    """Count false positives (predictions not near any known site) per 1000 km^2.

    Args:
        predicted_polygons: Each has 'centroid_lon', 'centroid_lat'.
        known_sites: Each has 'lon', 'lat'.
        total_area_km2: Total area scanned.
        match_distance_km: Max distance to match.

    Returns:
        FP rate per 1000 km^2.
    """
    if total_area_km2 <= 0:
        return 0.0

    match_deg = match_distance_km / 111.0
    fp_count = 0

    for pred in predicted_polygons:
        pred_pt = Point(pred["centroid_lon"], pred["centroid_lat"])
        is_tp = any(
            pred_pt.distance(Point(s["lon"], s["lat"])) < match_deg
            for s in known_sites
        )
        if not is_tp:
            fp_count += 1

    return fp_count / (total_area_km2 / 1000.0)


def calibration_bins(
    predictions: list[dict],
    known_sites: list[dict],
    n_bins: int = 10,
    match_distance_km: float = 5.0,
) -> dict:
    """Compute calibration: for each confidence bin, what fraction are true positives?

    Returns dict with 'bin_edges', 'bin_accuracy', 'bin_confidence', 'bin_count'.
    """
    match_deg = match_distance_km / 111.0
    scores = np.array([p["score"] for p in predictions])

    is_tp = np.zeros(len(predictions), dtype=bool)
    for i, pred in enumerate(predictions):
        pred_pt = Point(pred["centroid_lon"], pred["centroid_lat"])
        is_tp[i] = any(
            pred_pt.distance(Point(s["lon"], s["lat"])) < match_deg
            for s in known_sites
        )

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accuracy = []
    bin_confidence = []
    bin_count = []

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (scores >= lo) & (scores < hi)
        count = mask.sum()
        bin_count.append(int(count))
        if count > 0:
            bin_accuracy.append(float(is_tp[mask].mean()))
            bin_confidence.append(float(scores[mask].mean()))
        else:
            bin_accuracy.append(0.0)
            bin_confidence.append((lo + hi) / 2)

    return {
        "bin_edges": bin_edges.tolist(),
        "bin_accuracy": bin_accuracy,
        "bin_confidence": bin_confidence,
        "bin_count": bin_count,
    }
