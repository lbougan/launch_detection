"""Evidence endpoints: before/after composite thumbnails for detections."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import rasterio
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from rasterio.windows import from_bounds
from sqlalchemy import func
from sqlalchemy.orm import Session

from services.api.database import get_db
from services.api.models import Detection

router = APIRouter()


@router.get("/{detection_id}")
def get_evidence(
    detection_id: int,
    db: Session = Depends(get_db),
) -> dict:
    """Get evidence metadata for a detection (dates, bands, scores)."""
    row = (
        db.query(
            Detection.evidence,
            Detection.score,
            Detection.model_version,
            func.ST_AsGeoJSON(Detection.geom).label("geom_geojson"),
        )
        .filter(Detection.id == detection_id)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Detection not found")

    import json
    return {
        "detection_id": detection_id,
        "score": row.score,
        "model_version": row.model_version,
        "evidence": row.evidence or {},
        "geometry": json.loads(row.geom_geojson),
    }


@router.get("/{detection_id}/thumbnail")
def get_evidence_thumbnail(
    detection_id: int,
    width: int = Query(256, ge=64, le=1024),
    height: int = Query(256, ge=64, le=1024),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """Get a PNG thumbnail of the detection area from the source composite.

    Returns a false-color RGB thumbnail for visual inspection.
    """
    row = (
        db.query(Detection.evidence)
        .filter(Detection.id == detection_id)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Detection not found")

    evidence = row.evidence or {}
    composite_path = evidence.get("composite_path")
    bbox = evidence.get("bbox")

    if not composite_path or not bbox or not Path(composite_path).exists():
        raise HTTPException(status_code=404, detail="Evidence imagery not available")

    with rasterio.open(composite_path) as src:
        window = from_bounds(*bbox, transform=src.transform)
        rgb = src.read([3, 2, 1], window=window, out_shape=(3, height, width))

    rgb = rgb.astype(np.float32)
    for i in range(3):
        lo, hi = np.percentile(rgb[i], [2, 98])
        if hi - lo > 0:
            rgb[i] = np.clip((rgb[i] - lo) / (hi - lo) * 255, 0, 255)
    rgb = rgb.astype(np.uint8)

    from PIL import Image
    img = Image.fromarray(np.moveaxis(rgb, 0, -1))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")
