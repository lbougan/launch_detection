"""Health check endpoint."""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from services.api.database import get_db
from services.api.models import Detection
from services.api.schemas import HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter()


def _triton_ready() -> bool | None:
    """Check Triton server readiness, or None if not configured."""
    triton_url = os.environ.get("TRITON_URL")
    if not triton_url:
        return None
    try:
        import tritonclient.grpc as grpcclient

        client = grpcclient.InferenceServerClient(url=triton_url)
        ready = client.is_server_ready()
        client.close()
        return ready
    except Exception as exc:
        logger.debug("Triton health check failed: %s", exc)
        return False


@router.get("/health", response_model=HealthResponse)
def health_check(db: Session = Depends(get_db)) -> HealthResponse:
    count = db.query(func.count(Detection.id)).scalar() or 0

    triton_status = _triton_ready()
    status = "ok"
    if triton_status is False:
        status = "degraded"

    return HealthResponse(
        status=status,
        version="0.1.0",
        detection_count=count,
        triton_ready=triton_status,
    )
