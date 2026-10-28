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


def test_held_out_sites_are_far_from_each_other_and_from_training_sites():
    sites = {s["name"]: s for s in _load("known_sites.json")["sites"]}
    splits = _load("splits.json")
    for held in splits["val"] + splits["test"]:
        for other in splits["train"]:
            d = distance_m(
                sites[held]["lon"], sites[held]["lat"], sites[other]["lon"], sites[other]["lat"]
            )
            assert d > 50_000, f"{held} is within 50 km of training site {other}"
