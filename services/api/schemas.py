"""Pydantic schemas for API request/response models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class DetectionOut(BaseModel):
    id: int
    score: float
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
