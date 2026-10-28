import pytest

pytestmark = pytest.mark.api


def test_known_sites_listed(client, seeded):
    names = {s["name"] for s in client.get("/known-sites").json()}
    assert names == {"Planted", "Far"}
