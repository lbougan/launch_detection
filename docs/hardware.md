# Hardware notes

This section was moved out of the README in v0.2. The text under "Hardware requirements" is the
v0.1 text, unchanged; the two sections after it are new.

## Hardware requirements

The full pipeline (ingest + train + detect) is **not designed for a laptop**. While the API, web map and Docker services are lightweight, the data volumes and compute demands of ingestion and training require a proper workstation or cloud VM.

### Data volume estimates

The default AOI config defines four regions totalling ~33,000 km² of ground coverage. At 10 m GSD with up to 50 Sentinel-2 scenes per AOI over a 6-month window:

| Stage | Per AOI | 4 AOIs (total) |
|---|---|---|
| Raw scene downloads (6 bands, uint16, deflate) | 25–40 GB | **80–150 GB** |
| Temporal composites (float32, 6 bands) | 1–3 GB | ~5–10 GB |
| Tiles (256×256, 10 channels float32) + labels | 3–6 GB | ~10–20 GB |
| Predictions + PostGIS + MLflow artifacts | 1–2 GB | ~3–5 GB |
| **Total disk footprint** | | **~100–180 GB** |

### Compute bottlenecks

| Stage | What hurts | Why |
|---|---|---|
| **Ingestion** | Network I/O, disk | Downloads tens of GB from Planetary Computer via concurrent HTTP reads (4 scenes × 6 bands in parallel). |
| **Compositing** | RAM | Loads all scenes for an AOI into memory for pixel-wise median — the largest AOI (Baikonur, ~12,700 km²) can require 30–60 GB of RAM. |
| **Training** | GPU (or CPU time) | ResNet-34 UNet, 10-channel 256×256 tiles, batch 16, up to 50 epochs. On CPU this takes hours to days; on a mid-range GPU it finishes in under an hour. |
| **Inference** | GPU (or CPU time) | Sliding window over full composites produces thousands of 256×256 patches per AOI. |

### Minimum and recommended specs

| Resource | Minimum (will be slow) | Recommended |
|---|---|---|
| **GPU** | None (CPU-only, `gpus=0`) | NVIDIA GPU with ≥ 8 GB VRAM (e.g. RTX 3060 12 GB, T4, A10G) |
| **RAM** | 16 GB | 32–64 GB |
| **Disk** | 200 GB free (SSD) | 500 GB+ SSD |
| **CPU** | 4 cores | 8+ cores (helps with data loaders and parallel downloads) |
| **Network** | Broadband | ≥ 100 Mbps (large COG downloads from Planetary Computer) |

### Suggested cloud instances

| Provider | Instance | GPU | RAM | Notes |
|---|---|---|---|---|
| AWS | `g5.2xlarge` | A10G 24 GB | 32 GB | Good balance of GPU and RAM |
| GCP | `n1-standard-8` + T4 | T4 16 GB | 30 GB | Attach a 500 GB persistent SSD |
| Azure | `Standard_NC8as_T4_v3` | T4 16 GB | 56 GB | Generous RAM for compositing |

### What runs fine on a laptop

The **API + web map** are lightweight and work well on any machine. For local development without running the pipeline, use the minimal quick start (`--skip-ingest --skip-training`) with pre-computed data or an empty database.

## v0.2 AOIs

The estimates above were written for the four large v0.1 AOIs (about 34,000 km² measured from the bounding boxes in `data/manifests/aoi_config.json`). v0.2 keeps those
four and adds 13 compact AOIs (400 to 1,100 km² each) so that every one of the 18 known sites has
imagery: 17 AOIs, about 45,000 km² in total. The compact AOIs add roughly a third to the area, so
scale the disk figures accordingly, or cap downloads with
`python -m services.pipeline.flows ingest --max-scenes 20` (what `scripts/gpu_day.sh` does).
v0.2 scene crops are float32 reflectance rather than uint16, so the raw-download disk figures
above roughly double.

## Outcome of issue #1 (CPU-only runs are too slow)

[Issue #1](https://github.com/lbougan/launch_detection/issues/1) reported that the local
pipeline is too slow on an 8-core CPU-only machine. Decision: run training and large-area
inference on a remote GPU, serving the model through Triton, rather than tune the CPU path. The
CPU path stays correct and tested (the synthetic end-to-end test trains for two epochs on CPU in
about 25 s), but it is not optimised for real AOIs. `scripts/gpu_day.sh` is the runbook for the
remote GPU machine.
