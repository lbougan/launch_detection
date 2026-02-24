"""FastAPI application for serving launch-site detections."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.api.routes import detections, evidence, health, known_sites, tiles

app = FastAPI(
    title="Launch Site Detection API",
    description="Query and browse satellite-based launch site detections",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["health"])
app.include_router(detections.router, prefix="/detections", tags=["detections"])
app.include_router(known_sites.router, prefix="/known-sites", tags=["known sites"])
app.include_router(evidence.router, prefix="/evidence", tags=["evidence"])
app.include_router(tiles.router, prefix="/tiles", tags=["tiles"])
