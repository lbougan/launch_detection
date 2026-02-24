"""Vector tile endpoint for map rendering."""

from __future__ import annotations

import io

from fastapi import APIRouter, Depends
from fastapi.responses import Response
from sqlalchemy import text
from sqlalchemy.orm import Session

from services.api.database import get_db

router = APIRouter()


@router.get("/{z}/{x}/{y}.pbf")
def get_vector_tile(
    z: int,
    x: int,
    y: int,
    db: Session = Depends(get_db),
) -> Response:
    """Serve Mapbox Vector Tiles (MVT) for detections.

    Uses PostGIS ST_AsMVT to generate tiles on the fly.
    """
    query = text("""
        WITH bounds AS (
            SELECT ST_TileEnvelope(:z, :x, :y) AS geom
        ),
        mvt_data AS (
            SELECT
                ST_AsMVTGeom(d.geom, bounds.geom) AS geom,
                d.id,
                d.score,
                d.area_km2,
                d.compactness,
                d.model_version
            FROM detections d, bounds
            WHERE ST_Intersects(d.geom, bounds.geom)
              AND d.score >= 0.1
        )
        SELECT ST_AsMVT(mvt_data, 'detections') AS tile
        FROM mvt_data
    """)

    result = db.execute(query, {"z": z, "x": x, "y": y}).scalar()
    tile_bytes = bytes(result) if result else b""

    return Response(
        content=tile_bytes,
        media_type="application/vnd.mapbox-vector-tile",
        headers={"Cache-Control": "max-age=3600"},
    )
