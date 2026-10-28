from services.training.metrics import fp_per_1000km2, reliability_bins, site_recall_at_k

SITES = [{"name": "A", "lon": 0.0, "lat": 0.0}, {"name": "B", "lon": 10.0, "lat": 10.0}]


def _pred(lon, lat, prob, rank):
    return {"centroid_lon": lon, "centroid_lat": lat, "prob": prob, "rank_score": rank}


def test_recall_at_k_counts_sites_hit_in_top_k():
    preds = [_pred(0.01, 0.0, 0.9, 0.9), _pred(5, 5, 0.8, 0.8), _pred(10.01, 10.0, 0.7, 0.7)]
    assert site_recall_at_k(preds, SITES, k=1, match_m=5000) == 0.5
    assert site_recall_at_k(preds, SITES, k=3, match_m=5000) == 1.0


def test_fp_rate_per_1000km2():
    preds = [_pred(0.0, 0.0, 0.9, 0.9), _pred(5, 5, 0.8, 0.8), _pred(6, 6, 0.8, 0.8)]
    assert fp_per_1000km2(preds, SITES, scanned_km2=4000.0, match_m=5000) == 0.5


def test_reliability_bins_use_prob_not_rank_score():
    preds = [_pred(0.0, 0.0, 0.95, 0.1), _pred(5, 5, 0.95, 0.1)]
    bins = reliability_bins(preds, SITES, n_bins=10, match_m=5000)
    assert bins["bin_count"][9] == 2 and bins["bin_count"][0] == 0
    assert bins["bin_accuracy"][9] == 0.5
