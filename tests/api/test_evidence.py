import pytest

pytestmark = pytest.mark.api


def test_thumbnail_renders_png_for_both_modes(client, seeded):
    ids = [d["id"] for d in client.get("/detections").json()["detections"]]
    with_evidence = ids[0]  # highest score first = the planted one
    for mode in ("rgb", "nir"):
        r = client.get(
            f"/evidence/{with_evidence}/thumbnail",
            params={"mode": mode, "width": 64, "height": 64},
        )
        assert (
            r.status_code == 200
            and r.headers["content-type"] == "image/png"
            and r.content[:8] == b"\x89PNG\r\n\x1a\n"
        )
    assert client.get(f"/evidence/{ids[-1]}/thumbnail").status_code == 404, (
        "no composite_path -> 404"
    )
    r = client.get(f"/evidence/{with_evidence}/thumbnail", params={"mode": "x"})
    assert r.status_code == 422
