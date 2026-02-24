CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS imagery_catalog (
    id              SERIAL PRIMARY KEY,
    scene_id        TEXT NOT NULL UNIQUE,
    sensor          TEXT NOT NULL,
    acquired_date   TIMESTAMPTZ NOT NULL,
    bbox            GEOMETRY(Polygon, 4326) NOT NULL,
    cloud_cover_pct REAL,
    cog_path        TEXT NOT NULL,
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_imagery_bbox ON imagery_catalog USING GIST (bbox);
CREATE INDEX idx_imagery_date ON imagery_catalog (acquired_date);

CREATE TABLE IF NOT EXISTS tiles (
    id              SERIAL PRIMARY KEY,
    tile_id         TEXT NOT NULL UNIQUE,
    bbox            GEOMETRY(Polygon, 4326) NOT NULL,
    composite_window TEXT,
    image_path      TEXT NOT NULL,
    label_path      TEXT,
    split           TEXT DEFAULT 'train',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_tiles_bbox ON tiles USING GIST (bbox);

CREATE TABLE IF NOT EXISTS detections (
    id              SERIAL PRIMARY KEY,
    geom            GEOMETRY(Polygon, 4326) NOT NULL,
    score           REAL NOT NULL,
    model_version   TEXT NOT NULL,
    evidence        JSONB DEFAULT '{}',
    area_km2        REAL,
    compactness     REAL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_detections_geom ON detections USING GIST (geom);
CREATE INDEX idx_detections_score ON detections (score DESC);

CREATE TABLE IF NOT EXISTS known_sites (
    id              SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    country         TEXT,
    geom            GEOMETRY(Point, 4326) NOT NULL,
    buffer_geom     GEOMETRY(Polygon, 4326),
    source          TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX idx_known_sites_geom ON known_sites USING GIST (geom);
