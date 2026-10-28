"""API tests run against a real PostGIS (CI service container or `docker compose up postgres`)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import rasterio
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from libs.config import settings
from tests.fixtures.synthetic import make_synthetic_composite, site_lonlat

pytestmark = pytest.mark.api


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "infra" / "postgres-init" / "001_schema.sql"


def _db_available() -> bool:
    try:
        with create_engine(settings.database_url).connect() as conn:
            conn.execute(text("select 1"))
        return True
    except Exception:
        return False


def _ensure_database(name: str) -> None:
    """CREATE DATABASE `name` on the configured server unless it exists."""
    admin = create_engine(settings.database_url, isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            exists = conn.execute(
                text("select 1 from pg_database where datname = :n"), {"n": name}
            ).scalar()
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{name}"'))
    finally:
        admin.dispose()


@pytest.fixture(scope="session")
def engine():
    """Engine on `<postgres_db>_test`; the app and pipeline settings point there too.

    The configured database itself is never touched beyond the existence check.
    """
    if not _db_available():
        if os.environ.get("CI"):
            pytest.fail("PostGIS not reachable in CI")
        pytest.skip("PostGIS not reachable at settings.database_url")
    from services.api.database import get_engine

    test_db = f"{settings.postgres_db}_test"
    _ensure_database(test_db)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(settings, "postgres_db", test_db)
        get_engine.cache_clear()
        eng = create_engine(settings.database_url)
        with eng.begin() as conn:
            conn.execute(text(SCHEMA_PATH.read_text()))
        yield eng
        eng.dispose()
        get_engine.cache_clear()


@pytest.fixture
def clean_db(engine):
    with engine.begin() as conn:
        conn.execute(text("truncate detections, known_sites restart identity"))
    yield engine


@pytest.fixture
def seeded(clean_db, tmp_path):
    """Three detections (one with real evidence imagery) and two known sites."""
    comp_path, site = make_synthetic_composite(
        tmp_path / "c.tif", height=320, width=320, site_size_px=24
    )
    lon, lat = site_lonlat(site)
    with rasterio.open(comp_path) as src:
        b = src.bounds
        from rasterio.warp import transform_bounds

        bbox_ll = transform_bounds(src.crs, "EPSG:4326", b.left, b.bottom, b.right, b.top)
    evidence = {
        "prob": 0.9,
        "rank_breakdown": {},
        "composite_path": str(comp_path),
        "bbox": list(bbox_ll),
        "model_version": "test",
        "scan_date": "2026-10-28",
    }
    rows = [
        (lon, lat, 0.9, 0.95, evidence),
        (lon + 0.5, lat, 0.4, 0.5, {}),
        (lon + 1.0, lat, 0.1, 0.2, {}),
    ]
    with clean_db.begin() as conn:
        for x, y, prob, score, ev in rows:
            conn.execute(
                text(
                    "insert into detections "
                    "(geom, score, prob, model_version, area_km2, compactness, evidence) "
                    "values (ST_Buffer(ST_SetSRID(ST_MakePoint(:x,:y),4326)::geography, 500)"
                    "::geometry, :s, :p, 'test', 0.8, 0.9, CAST(:e AS jsonb))"
                ),
                {"x": x, "y": y, "s": score, "p": prob, "e": json.dumps(ev)},
            )
        conn.execute(
            text(
                "insert into known_sites (name, country, geom) values "
                "('Planted', 'XX', ST_SetSRID(ST_MakePoint(:x,:y),4326)), "
                "('Far', 'YY', ST_SetSRID(ST_MakePoint(0,0),4326))"
            ),
            {"x": lon, "y": lat},
        )
    return {"lon": lon, "lat": lat, "composite": comp_path, "engine": clean_db}


@pytest.fixture
def client(engine):
    from services.api.main import app

    return TestClient(app)
