import io
import json

import numpy as np
import pytest
import rasterio
from PIL import Image
from sqlalchemy import text

pytestmark = pytest.mark.api


def _png_array(content: bytes) -> np.ndarray:
    assert content[:8] == b"\x89PNG\r\n\x1a\n"
    return np.asarray(Image.open(io.BytesIO(content)).convert("RGB"))


def test_thumbnail_renders_png_for_both_modes(client, seeded):
    ids = [d["id"] for d in client.get("/detections").json()["detections"]]
    with_evidence = ids[0]  # highest score first = the planted one
    arrays = {}
    for mode in ("rgb", "nir"):
        r = client.get(
            f"/evidence/{with_evidence}/thumbnail",
            params={"mode": mode, "width": 64, "height": 64},
        )
        assert r.status_code == 200 and r.headers["content-type"] == "image/png"
        arrays[mode] = _png_array(r.content)
        assert arrays[mode].shape == (64, 64, 3) and arrays[mode].any(), f"{mode} is all zero"
    assert not np.array_equal(arrays["rgb"], arrays["nir"])
    assert client.get(f"/evidence/{ids[-1]}/thumbnail").status_code == 404, (
        "no composite_path -> 404"
    )
    r = client.get(f"/evidence/{with_evidence}/thumbnail", params={"mode": "x"})
    assert r.status_code == 422


def test_thumbnail_404_on_composite_without_bandspec(client, seeded, tmp_path):
    untagged = tmp_path / "untagged.tif"
    with rasterio.open(seeded["composite"]) as src:
        profile, data = src.profile, src.read()
        bbox = rasterio.warp.transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    with rasterio.open(untagged, "w", **profile) as dst:
        dst.write(data)
    ev = {"composite_path": str(untagged), "bbox": list(bbox)}
    with seeded["engine"].begin() as conn:
        det_id = conn.execute(
            text(
                "insert into detections (geom, score, model_version, evidence) values "
                "(ST_Buffer(ST_SetSRID(ST_MakePoint(0,0),4326), 0.01), 0.3, 't', "
                "CAST(:e AS jsonb)) returning id"
            ),
            {"e": json.dumps(ev)},
        ).scalar_one()
    r = client.get(f"/evidence/{det_id}/thumbnail")
    assert r.status_code == 404 and "band spec" in r.json()["detail"]
