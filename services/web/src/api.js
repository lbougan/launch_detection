const API_BASE = "";

export async function fetchDetections(bbox, minScore = 0.3, limit = 500) {
  const [minLon, minLat, maxLon, maxLat] = bbox;
  const params = new URLSearchParams({
    min_lon: minLon,
    min_lat: minLat,
    max_lon: maxLon,
    max_lat: maxLat,
    min_score: minScore,
    limit,
  });
  const res = await fetch(`${API_BASE}/detections?${params}`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function fetchDetectionDetail(id) {
  const res = await fetch(`${API_BASE}/detections/${id}`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function fetchKnownSites() {
  const res = await fetch(`${API_BASE}/known-sites`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export async function fetchEvidence(id) {
  const res = await fetch(`${API_BASE}/evidence/${id}`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export function evidenceThumbnailUrl(id, width = 256, height = 256) {
  return `${API_BASE}/evidence/${id}/thumbnail?width=${width}&height=${height}`;
}

export function vectorTileUrl() {
  return `${API_BASE}/tiles/{z}/{x}/{y}.pbf`;
}
