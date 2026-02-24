"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from services.api.database import get_db
from services.api.models import Detection
from services.api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check(db: Session = Depends(get_db)) -> HealthResponse:
    count = db.query(func.count(Detection.id)).scalar() or 0
    return HealthResponse(status="ok", version="0.1.0", detection_count=count)
