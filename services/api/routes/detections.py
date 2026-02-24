"""Detections query endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from geoalchemy2.functions import ST_AsGeoJSON, ST_Envelope, ST_MakeEnvelope, ST_X, ST_Y
from sqlalchemy import func
from sqlalchemy.orm import Session

from services.api.database import get_db
from services.api.models import Detection
from services.api.schemas import DetectionDetail, DetectionListResponse, DetectionOut

router = APIRouter()


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
    """Query detections within a bounding box, filtered by score."""
    bbox = ST_MakeEnvelope(min_lon, min_lat, max_lon, max_lat, 4326)

    query = (
        db.query(
            Detection.id,
            Detection.score,
            Detection.model_version,
            Detection.area_km2,
            Detection.compactness,
            Detection.evidence,
            Detection.created_at,
            func.ST_X(func.ST_Centroid(Detection.geom)).label("centroid_lon"),
            func.ST_Y(func.ST_Centroid(Detection.geom)).label("centroid_lat"),
            func.ST_AsGeoJSON(func.ST_Envelope(Detection.geom)).label("bbox_geojson"),
        )
        .filter(Detection.geom.ST_Intersects(bbox))
        .filter(Detection.score >= min_score)
        .order_by(Detection.score.desc())
        .offset(offset)
        .limit(limit)
    )

    rows = query.all()
    detections = []
    for row in rows:
        import json
        bbox_geo = json.loads(row.bbox_geojson)
        coords = bbox_geo["coordinates"][0]
        flat_bbox = [coords[0][0], coords[0][1], coords[2][0], coords[2][1]]
        detections.append(
            DetectionOut(
                id=row.id,
                score=row.score,
                model_version=row.model_version,
                area_km2=row.area_km2,
                compactness=row.compactness,
                evidence=row.evidence or {},
                centroid_lon=row.centroid_lon,
                centroid_lat=row.centroid_lat,
                bbox=flat_bbox,
                created_at=row.created_at,
            )
        )

    total = (
        db.query(func.count(Detection.id))
        .filter(Detection.geom.ST_Intersects(bbox))
        .filter(Detection.score >= min_score)
        .scalar()
    )

    return DetectionListResponse(count=total or 0, detections=detections)


@router.get("/{detection_id}", response_model=DetectionDetail)
def get_detection(detection_id: int, db: Session = Depends(get_db)) -> DetectionDetail:
    """Get a single detection with full geometry."""
    row = (
        db.query(
            Detection.id,
            Detection.score,
            Detection.model_version,
            Detection.area_km2,
            Detection.compactness,
            Detection.evidence,
            Detection.created_at,
            func.ST_X(func.ST_Centroid(Detection.geom)).label("centroid_lon"),
            func.ST_Y(func.ST_Centroid(Detection.geom)).label("centroid_lat"),
            func.ST_AsGeoJSON(func.ST_Envelope(Detection.geom)).label("bbox_geojson"),
            func.ST_AsGeoJSON(Detection.geom).label("geom_geojson"),
        )
        .filter(Detection.id == detection_id)
        .first()
    )

    if not row:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Detection not found")

    import json
    bbox_geo = json.loads(row.bbox_geojson)
    coords = bbox_geo["coordinates"][0]
    flat_bbox = [coords[0][0], coords[0][1], coords[2][0], coords[2][1]]

    return DetectionDetail(
        id=row.id,
        score=row.score,
        model_version=row.model_version,
        area_km2=row.area_km2,
        compactness=row.compactness,
        evidence=row.evidence or {},
        centroid_lon=row.centroid_lon,
        centroid_lat=row.centroid_lat,
        bbox=flat_bbox,
        created_at=row.created_at,
        geojson=json.loads(row.geom_geojson),
    )
