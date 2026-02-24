"""Known launch sites endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from services.api.database import get_db
from services.api.models import KnownSite
from services.api.schemas import KnownSiteOut

router = APIRouter()


@router.get("", response_model=list[KnownSiteOut])
def list_known_sites(db: Session = Depends(get_db)) -> list[KnownSiteOut]:
    """List all known launch sites (for map display and evaluation)."""
    rows = db.query(
        KnownSite.id,
        KnownSite.name,
        KnownSite.country,
        func.ST_X(KnownSite.geom).label("lon"),
        func.ST_Y(KnownSite.geom).label("lat"),
    ).all()

    return [
        KnownSiteOut(id=r.id, name=r.name, country=r.country, lon=r.lon, lat=r.lat)
        for r in rows
    ]
