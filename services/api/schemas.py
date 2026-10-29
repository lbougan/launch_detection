"""Pydantic schemas for API request/response models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

# Evidence keys that stay on the server: filesystem paths of the API host.
PRIVATE_EVIDENCE_KEYS = frozenset({"composite_path"})


def public_evidence(evidence: dict | None) -> dict:
    """The evidence dict as sent over the wire, without server-side keys."""
    return {k: v for k, v in (evidence or {}).items() if k not in PRIVATE_EVIDENCE_KEYS}


class DetectionOut(BaseModel):
    id: int
    score: float  # rank score
    prob: float | None = None  # mean model probability
    model_version: str
    area_km2: float | None = None
    compactness: float | None = None
    evidence: dict = Field(default_factory=dict)
    centroid_lon: float
    centroid_lat: float
    bbox: list[float]
    created_at: datetime

    model_config = {"from_attributes": True}


class DetectionDetail(DetectionOut):
    geojson: dict


class DetectionListResponse(BaseModel):
    count: int
    detections: list[DetectionOut]


class KnownSiteOut(BaseModel):
    id: int
    name: str
    country: str | None = None
    lon: float
    lat: float

    model_config = {"from_attributes": True}


class HealthResponse(BaseModel):
    status: str
    version: str
    detection_count: int
    triton_ready: bool | None = None
