"""store_detections_task / seed_known_sites_task against PostGIS."""

import json

import pytest
from sqlalchemy import text

from libs.config import settings
from services.pipeline.tasks import seed_known_sites_task, store_detections_task

pytestmark = pytest.mark.api


def _square(lon: float, lat: float, d: float = 0.01) -> dict:
    ring = [[lon, lat], [lon + d, lat], [lon + d, lat + d], [lon, lat + d], [lon, lat]]
    return {"type": "Polygon", "coordinates": [ring]}


def test_store_writes_prob_rank_score_and_renderable_evidence(clean_db):
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": _square(10.0, 20.0),
                "properties": {
                    "prob": 0.7,
                    "rank_score": 0.85,
                    "area_km2": 1.2,
                    "compactness": 0.6,
                    "rank_breakdown": {"prob": 0.7, "area": 1.0},
                },
            }
        ],
    }
    n = store_detections_task.fn(fc, "v-test", composite_path="/data/c.tif", scan_date="2026-10-28")
    assert n == 1
    with clean_db.connect() as conn:
        row = conn.execute(text("select score, prob, evidence from detections")).one()
    assert row.score == pytest.approx(0.85) and row.prob == pytest.approx(0.7)
    ev = row.evidence
    assert ev["prob"] == 0.7 and ev["rank_breakdown"] == {"prob": 0.7, "area": 1.0}
    assert ev["composite_path"] == "/data/c.tif" and ev["model_version"] == "v-test"
    assert ev["scan_date"] == "2026-10-28"
    assert ev["bbox"] == pytest.approx([10.0, 20.0, 10.01, 20.01])


def test_seed_known_sites_is_idempotent(clean_db, tmp_path):
    p = tmp_path / "sites.json"
    sites = [
        {"name": "A", "lat": 1.0, "lon": 2.0, "country": "XX"},
        {"name": "B", "lat": 3.0, "lon": 4.0},
    ]
    p.write_text(json.dumps({"sites": sites}))
    assert seed_known_sites_task.fn(str(p), db_url=settings.database_url) == 2
    assert seed_known_sites_task.fn(str(p)) == 0
    with clean_db.connect() as conn:
        rows = conn.execute(
            text("select name, country, ST_X(geom) x, ST_Y(geom) y from known_sites order by name")
        ).all()
    assert [(r.name, r.country, r.x, r.y) for r in rows] == [
        ("A", "XX", 2.0, 1.0),
        ("B", None, 4.0, 3.0),
    ]
