"""Vector tile endpoint for map rendering."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
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
    min_score: float = Query(0.1, ge=0, le=1),
    db: Session = Depends(get_db),
) -> Response:
    """Serve Mapbox Vector Tiles (MVT) for detections.

    Uses PostGIS ST_AsMVT to generate tiles on the fly; `min_score` filters on the
    rank score so the map slider and the tiles agree.
    """
    query = text("""
        WITH bounds AS (
            SELECT ST_TileEnvelope(:z, :x, :y) AS geom_3857,
                   ST_Transform(ST_TileEnvelope(:z, :x, :y), 4326) AS geom_4326
        ),
        mvt_data AS (
            SELECT
                -- A site polygon is sub-pixel at low zoom and ST_AsMVTGeom drops it;
                -- fall back to its centroid so the site still shows as a point.
                -- The layer therefore mixes polygons and centroid points (small sites at
                -- low zoom). The bundled web map draws only the polygons from this layer
                -- (zoom >= 9) and takes centroids from /detections; a map built on tiles
                -- alone needs a circle layer filtered on geometry-type Point as well.
                COALESCE(
                    ST_AsMVTGeom(ST_Transform(d.geom, 3857), bounds.geom_3857),
                    ST_AsMVTGeom(ST_Transform(ST_Centroid(d.geom), 3857), bounds.geom_3857)
                ) AS geom,
                d.id,
                d.score,
                d.prob,
                d.area_km2,
                d.compactness,
                d.model_version
            FROM detections d, bounds
            WHERE ST_Intersects(d.geom, bounds.geom_4326)
              AND d.score >= :min_score
        )
        SELECT ST_AsMVT(mvt_data, 'detections') AS tile
        FROM mvt_data
    """)

    result = db.execute(query, {"z": z, "x": x, "y": y, "min_score": min_score}).scalar()
    tile_bytes = bytes(result) if result else b""

    return Response(
        content=tile_bytes,
        media_type="application/vnd.mapbox-vector-tile",
        headers={"Cache-Control": "max-age=3600"},
    )
