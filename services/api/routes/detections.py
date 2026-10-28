"""Detections query endpoints."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Query
from geoalchemy2.functions import ST_MakeEnvelope
from sqlalchemy import func
from sqlalchemy.orm import Session

from services.api.database import get_db
from services.api.models import Detection
from services.api.schemas import DetectionDetail, DetectionListResponse, DetectionOut

router = APIRouter()

_COLUMNS = (
    Detection.id,
    Detection.score,
    Detection.prob,
    Detection.model_version,
    Detection.area_km2,
    Detection.compactness,
    Detection.evidence,
    Detection.created_at,
    func.ST_X(func.ST_Centroid(Detection.geom)).label("centroid_lon"),
    func.ST_Y(func.ST_Centroid(Detection.geom)).label("centroid_lat"),
    func.ST_XMin(Detection.geom).label("xmin"),
    func.ST_YMin(Detection.geom).label("ymin"),
    func.ST_XMax(Detection.geom).label("xmax"),
    func.ST_YMax(Detection.geom).label("ymax"),
)


def _row_to_out(row) -> DetectionOut:
    return DetectionOut(
        id=row.id,
        score=row.score,
        prob=row.prob,
        model_version=row.model_version,
        area_km2=row.area_km2,
        compactness=row.compactness,
        evidence=row.evidence or {},
        centroid_lon=row.centroid_lon,
        centroid_lat=row.centroid_lat,
        bbox=[row.xmin, row.ymin, row.xmax, row.ymax],
        created_at=row.created_at,
    )


@router.get("", response_model=DetectionListResponse)
def list_detections(
    min_lon: float = Query(-180),
    min_lat: float = Query(-90),
    max_lon: float = Query(180),
    max_lat: float = Query(90),
    min_score: float = Query(0.0, ge=0, le=1),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> DetectionListResponse:
    """Query detections within a bounding box, filtered by rank score."""
    bbox = ST_MakeEnvelope(min_lon, min_lat, max_lon, max_lat, 4326)
    filters = (Detection.geom.ST_Intersects(bbox), Detection.score >= min_score)

    rows = (
        db.query(*_COLUMNS)
        .filter(*filters)
        .order_by(Detection.score.desc(), Detection.id)
        .offset(offset)
        .limit(limit)
        .all()
    )
    total = db.query(func.count(Detection.id)).filter(*filters).scalar()
    return DetectionListResponse(count=total or 0, detections=[_row_to_out(r) for r in rows])


@router.get("/{detection_id}", response_model=DetectionDetail)
def get_detection(detection_id: int, db: Session = Depends(get_db)) -> DetectionDetail:
    """Get a single detection with full geometry."""
    row = (
        db.query(*_COLUMNS, func.ST_AsGeoJSON(Detection.geom).label("geom_geojson"))
        .filter(Detection.id == detection_id)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Detection not found")
    return DetectionDetail(**_row_to_out(row).model_dump(), geojson=json.loads(row.geom_geojson))
