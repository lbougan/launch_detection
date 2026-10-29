import pytest

pytestmark = pytest.mark.api


def test_mvt_honours_min_score(client, seeded):
    empty = client.get("/tiles/0/0/0.pbf", params={"min_score": 0.99})
    some = client.get("/tiles/0/0/0.pbf", params={"min_score": 0.1})
    assert empty.status_code == 200 and some.status_code == 200
    assert len(some.content) > len(empty.content)
    assert some.headers["content-type"] == "application/vnd.mapbox-vector-tile"
