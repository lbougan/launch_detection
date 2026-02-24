"""SQLAlchemy ORM models for the API."""

from __future__ import annotations

from datetime import datetime

from geoalchemy2 import Geometry
from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB

from services.api.database import Base


class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    geom = Column(Geometry("POLYGON", srid=4326), nullable=False)
    score = Column(Float, nullable=False)
    model_version = Column(String, nullable=False)
    evidence = Column(JSONB, default={})
    area_km2 = Column(Float)
    compactness = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class KnownSite(Base):
    __tablename__ = "known_sites"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    country = Column(String)
    geom = Column(Geometry("POINT", srid=4326), nullable=False)
    buffer_geom = Column(Geometry("POLYGON", srid=4326))
    source = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class ImageryCatalog(Base):
    __tablename__ = "imagery_catalog"

    id = Column(Integer, primary_key=True, index=True)
    scene_id = Column(String, unique=True, nullable=False)
    sensor = Column(String, nullable=False)
    acquired_date = Column(DateTime, nullable=False)
    bbox = Column(Geometry("POLYGON", srid=4326), nullable=False)
    cloud_cover_pct = Column(Float)
    cog_path = Column(String, nullable=False)
    extra_metadata = Column("metadata", JSONB, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
