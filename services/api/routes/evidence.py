"""Evidence endpoints: before/after composite thumbnails for detections."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from PIL import Image
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from sqlalchemy import func
from sqlalchemy.orm import Session

from libs.geo.bandspec import read_bandspec
from services.api.database import get_db
from services.api.models import Detection
from services.api.schemas import public_evidence

router = APIRouter()

# Display bands per mode, by BandSpec name: true colour, and NIR false colour
# (vegetation red, bare concrete/sand pale).
MODE_BANDS: dict[str, tuple[str, str, str]] = {
    "rgb": ("red", "green", "blue"),
    "nir": ("nir", "red", "green"),
}


def stretch_to_uint8(bands: np.ndarray) -> np.ndarray:
    """Per-band 2/98 percentile stretch ignoring NaN; NaN and flat bands become 0."""
    out = np.zeros(bands.shape, dtype=np.uint8)
    for i, band in enumerate(bands.astype(np.float32)):
        finite = np.isfinite(band)
        if not finite.any():
            continue
        lo, hi = np.nanpercentile(band[finite], [2, 98])
        if hi - lo <= 0:
            continue
        scaled = np.clip((band - lo) / (hi - lo) * 255.0, 0, 255)
        out[i] = np.where(finite, scaled, 0).astype(np.uint8)
    return out


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

    return {
        "detection_id": detection_id,
        "score": row.score,
        "model_version": row.model_version,
        "evidence": public_evidence(row.evidence),
        "geometry": json.loads(row.geom_geojson),
    }


@router.get("/{detection_id}/thumbnail")
def get_evidence_thumbnail(
    detection_id: int,
    mode: Literal["rgb", "nir"] = Query("rgb"),
    width: int = Query(256, ge=64, le=1024),
    height: int = Query(256, ge=64, le=1024),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """PNG of the detection's evidence bbox from the source composite.

    `mode=rgb` is true colour; `mode=nir` is NIR/red/green false colour.
    """
    row = db.query(Detection.evidence).filter(Detection.id == detection_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Detection not found")

    evidence = row.evidence or {}
    composite_path = evidence.get("composite_path")
    bbox = evidence.get("bbox")
    if not composite_path or not bbox or not Path(composite_path).exists():
        raise HTTPException(status_code=404, detail="Evidence imagery not available")

    with rasterio.open(composite_path) as src:
        try:
            names = read_bandspec(src).names
        except ValueError as exc:
            raise HTTPException(status_code=404, detail="Composite has no band spec") from exc
        missing = [b for b in MODE_BANDS[mode] if b not in names]
        if missing:
            raise HTTPException(
                status_code=404, detail=f"Composite lacks bands {missing} for mode={mode}"
            )
        indexes = [names.index(b) + 1 for b in MODE_BANDS[mode]]
        # evidence bbox is EPSG:4326; the composite is in its own (UTM) CRS.
        native = transform_bounds("EPSG:4326", src.crs, *bbox)
        window = from_bounds(*native, transform=src.transform)
        masked = src.read(
            indexes,
            window=window,
            out_shape=(3, height, width),
            boundless=True,
            masked=True,
            resampling=Resampling.bilinear,
        )
        bands = masked.astype(np.float32).filled(np.nan)

    img = Image.fromarray(np.moveaxis(stretch_to_uint8(bands), 0, -1))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
