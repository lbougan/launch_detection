import maplibregl from "maplibre-gl";
import {
  fetchDetections,
  fetchDetectionDetail,
  fetchKnownSites,
  fetchEvidence,
  evidenceThumbnailUrl,
  vectorTileUrl,
} from "./api.js";

const map = new maplibregl.Map({
  container: "map",
  style: {
    version: 8,
    glyphs: "https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf",
    sources: {
      "carto-dark": {
        type: "raster",
        tiles: [
          "https://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png",
        ],
        tileSize: 256,
        attribution: "&copy; CARTO &copy; OSM contributors",
      },
    },
    layers: [
      {
        id: "carto-dark-layer",
        type: "raster",
        source: "carto-dark",
        minzoom: 0,
        maxzoom: 19,
      },
    ],
  },
  center: [0, 25],
  zoom: 2,
});

map.addControl(new maplibregl.NavigationControl(), "top-left");

const scoreSlider = document.getElementById("score-slider");
const scoreValue = document.getElementById("score-value");
const toggleKnown = document.getElementById("toggle-known");
const toggleHeatmap = document.getElementById("toggle-heatmap");
const panel = document.getElementById("panel");
const panelContent = document.getElementById("panel-content");
const panelClose = document.getElementById("panel-close");

let currentMinScore = 0.3;

scoreSlider.addEventListener("input", (e) => {
  currentMinScore = e.target.value / 100;
  scoreValue.textContent = currentMinScore.toFixed(2);
  loadDetections();
});

panelClose.addEventListener("click", () => panel.classList.add("hidden"));

toggleKnown.addEventListener("change", (e) => {
  const vis = e.target.checked ? "visible" : "none";
  if (map.getLayer("known-sites-layer"))
    map.setLayoutProperty("known-sites-layer", "visibility", vis);
  if (map.getLayer("known-sites-labels"))
    map.setLayoutProperty("known-sites-labels", "visibility", vis);
});

toggleHeatmap.addEventListener("change", (e) => {
  if (map.getLayer("detections-heat"))
    map.setLayoutProperty(
      "detections-heat",
      "visibility",
      e.target.checked ? "visible" : "none"
    );
  if (map.getLayer("detections-circles"))
    map.setLayoutProperty(
      "detections-circles",
      "visibility",
      e.target.checked ? "none" : "visible"
    );
});

map.on("load", async () => {
  addVectorTileSource();
  await loadKnownSites();
  await loadDetections();

  map.on("click", "detections-circles", onDetectionClick);
  map.on("mouseenter", "detections-circles", () => {
    map.getCanvas().style.cursor = "pointer";
  });
  map.on("mouseleave", "detections-circles", () => {
    map.getCanvas().style.cursor = "";
  });
});

function addVectorTileSource() {
  map.addSource("detections-mvt", {
    type: "vector",
    tiles: [vectorTileUrl()],
    minzoom: 0,
    maxzoom: 14,
  });
}

async function loadKnownSites() {
  try {
    const sites = await fetchKnownSites();
    const geojson = {
      type: "FeatureCollection",
      features: sites.map((s) => ({
        type: "Feature",
        geometry: { type: "Point", coordinates: [s.lon, s.lat] },
        properties: { name: s.name, country: s.country },
      })),
    };

    map.addSource("known-sites", { type: "geojson", data: geojson });
    map.addLayer({
      id: "known-sites-layer",
      type: "circle",
      source: "known-sites",
      paint: {
        "circle-radius": 7,
        "circle-color": "#22d3ee",
        "circle-stroke-width": 2,
        "circle-stroke-color": "#0e7490",
        "circle-opacity": 0.85,
      },
    });
    map.addLayer({
      id: "known-sites-labels",
      type: "symbol",
      source: "known-sites",
      layout: {
        "text-field": ["get", "name"],
        "text-size": 11,
        "text-offset": [0, 1.5],
        "text-anchor": "top",
      },
      paint: {
        "text-color": "#a5f3fc",
        "text-halo-color": "#0a0a0f",
        "text-halo-width": 1.5,
      },
    });
  } catch (err) {
    console.warn("Could not load known sites:", err);
  }
}

async function loadDetections() {
  try {
    const bounds = map.getBounds();
    const bbox = [
      bounds.getWest(),
      bounds.getSouth(),
      bounds.getEast(),
      bounds.getNorth(),
    ];
    const data = await fetchDetections(bbox, currentMinScore);

    const geojson = {
      type: "FeatureCollection",
      features: data.detections.map((d) => ({
        type: "Feature",
        geometry: {
          type: "Point",
          coordinates: [d.centroid_lon, d.centroid_lat],
        },
        properties: {
          id: d.id,
          score: d.score,
          area_km2: d.area_km2,
          model_version: d.model_version,
        },
      })),
    };

    if (map.getSource("detections-geojson")) {
      map.getSource("detections-geojson").setData(geojson);
    } else {
      map.addSource("detections-geojson", { type: "geojson", data: geojson });

      map.addLayer({
        id: "detections-heat",
        type: "heatmap",
        source: "detections-geojson",
        layout: { visibility: "none" },
        paint: {
          "heatmap-weight": ["get", "score"],
          "heatmap-intensity": 1,
          "heatmap-color": [
            "interpolate",
            ["linear"],
            ["heatmap-density"],
            0, "rgba(0,0,0,0)",
            0.2, "#1e1b4b",
            0.4, "#4338ca",
            0.6, "#7c3aed",
            0.8, "#c084fc",
            1, "#f0abfc",
          ],
          "heatmap-radius": 30,
        },
      });

      map.addLayer({
        id: "detections-circles",
        type: "circle",
        source: "detections-geojson",
        paint: {
          "circle-radius": [
            "interpolate",
            ["linear"],
            ["get", "score"],
            0, 4,
            0.5, 7,
            1, 12,
          ],
          "circle-color": [
            "interpolate",
            ["linear"],
            ["get", "score"],
            0, "#374151",
            0.3, "#f59e0b",
            0.6, "#ef4444",
            0.9, "#dc2626",
          ],
          "circle-stroke-width": 1.5,
          "circle-stroke-color": "#1f2937",
          "circle-opacity": 0.9,
        },
      });
    }
  } catch (err) {
    console.warn("Could not load detections:", err);
  }
}

async function onDetectionClick(e) {
  const feature = e.features[0];
  const id = feature.properties.id;

  try {
    const detail = await fetchDetectionDetail(id);
    const evidence = await fetchEvidence(id).catch(() => null);

    panelContent.innerHTML = `
      <h2>Detection #${detail.id}</h2>
      <div class="field"><span class="label">Score</span><span class="value">${detail.score.toFixed(3)}</span></div>
      <div class="field"><span class="label">Area</span><span class="value">${detail.area_km2?.toFixed(2) ?? "—"} km²</span></div>
      <div class="field"><span class="label">Compactness</span><span class="value">${detail.compactness?.toFixed(3) ?? "—"}</span></div>
      <div class="field"><span class="label">Model</span><span class="value">${detail.model_version}</span></div>
      <div class="field"><span class="label">Detected</span><span class="value">${new Date(detail.created_at).toLocaleDateString()}</span></div>
      <div class="field"><span class="label">Center</span><span class="value">${detail.centroid_lat.toFixed(4)}, ${detail.centroid_lon.toFixed(4)}</span></div>
      <div class="thumbnail">
        <img src="${evidenceThumbnailUrl(id)}" alt="Evidence thumbnail" onerror="this.style.display='none'" />
      </div>
    `;
    panel.classList.remove("hidden");
  } catch (err) {
    console.error("Failed to load detection detail:", err);
  }
}

map.on("moveend", loadDetections);
