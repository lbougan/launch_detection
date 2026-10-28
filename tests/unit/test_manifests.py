import json
from pathlib import Path

from libs.geo.projection import distance_m

MANIFESTS = Path("data/manifests")


def _load(name):
    return json.loads((MANIFESTS / name).read_text())


def test_every_known_site_is_inside_an_aoi():
    sites = _load("known_sites.json")["sites"]
    aois = _load("aoi_config.json")["aois"]
    uncovered = [
        s["name"]
        for s in sites
        if not any(
            a["bbox"][0] <= s["lon"] <= a["bbox"][2] and a["bbox"][1] <= s["lat"] <= a["bbox"][3]
            for a in aois
        )
    ]
    assert uncovered == [], f"sites without imagery: {uncovered}"


def test_splits_cover_all_sites_exactly_once():
    sites = {s["name"] for s in _load("known_sites.json")["sites"]}
    splits = _load("splits.json")
    assigned = splits["train"] + splits["val"] + splits["test"]
    assert sorted(assigned) == sorted(sites)
    assert len(assigned) == len(set(assigned))
    assert splits["test"] == [
        "Satish Dhawan Space Centre",
        "Tanegashima Space Center",
        "Guiana Space Centre",
    ]
    assert splits["val"] == [
        "Xichang Satellite Launch Center",
        "Plesetsk Cosmodrome",
        "Sohae Satellite Launching Station",
    ]


def _dist(sites, a, b):
    return distance_m(sites[a]["lon"], sites[a]["lat"], sites[b]["lon"], sites[b]["lat"])


def test_held_out_sites_are_far_from_each_other_and_from_training_sites():
    sites = {s["name"]: s for s in _load("known_sites.json")["sites"]}
    splits = _load("splits.json")
    threshold = max(50_000, 2 * splits["exclusion_km"] * 1000)
    held_out = splits["val"] + splits["test"]
    for held in held_out:
        for other in splits["train"]:
            assert _dist(sites, held, other) > threshold, (
                f"{held} is too close to training site {other}"
            )
    for v in splits["val"]:
        for t in splits["test"]:
            assert _dist(sites, v, t) > threshold, f"val site {v} is too close to test site {t}"


def test_each_aoi_contains_sites_of_one_split_only():
    sites = _load("known_sites.json")["sites"]
    splits = _load("splits.json")
    site_split = {n: s for s in ("train", "val", "test") for n in splits[s]}
    for a in _load("aoi_config.json")["aois"]:
        inside = {
            site_split[s["name"]]
            for s in sites
            if a["bbox"][0] <= s["lon"] <= a["bbox"][2] and a["bbox"][1] <= s["lat"] <= a["bbox"][3]
        }
        assert len(inside) <= 1, f"AOI {a['name']} mixes splits: {inside}"
