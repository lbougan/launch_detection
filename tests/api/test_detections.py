import pytest

pytestmark = pytest.mark.api


def test_list_filters_by_bbox_and_min_score(client, seeded):
    r = client.get(
        "/detections",
        params={
            "min_lon": seeded["lon"] - 0.1,
            "max_lon": seeded["lon"] + 0.1,
            "min_lat": seeded["lat"] - 0.1,
            "max_lat": seeded["lat"] + 0.1,
        },
    )
    assert r.status_code == 200 and r.json()["count"] == 1
    d = r.json()["detections"][0]
    assert d["prob"] == 0.9 and d["score"] == 0.95
    assert client.get("/detections", params={"min_score": 0.6}).json()["count"] == 1
    assert client.get("/detections").json()["count"] == 3


def test_detail_has_geojson(client, seeded):
    first = client.get("/detections").json()["detections"][0]["id"]
    r = client.get(f"/detections/{first}")
    assert r.status_code == 200 and r.json()["geojson"]["type"] == "Polygon"
    assert client.get("/detections/999999").status_code == 404
